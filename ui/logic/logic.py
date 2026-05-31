import os
import re
import unicodedata
import streamlit as st
import pandas as pd
from datetime import datetime

from langchain_gigachat import GigaChat
from chains.langchain_chains import (
    EntityParserChain, InstructionGeneratorChain, ChatChain, ConversationManager
)
from agents.langchain_agents import create_registrar_agent, create_consultant_agent
from utils.audio_utils import text_to_speech, get_audio_html_player
from rag_data.build_knowledge_base import search

# ══════════════════════════════════════════════════════════════════════════════
# Бизнес-логика: обработка события
# ══════════════════════════════════════════════════════════════════════════════

# Ключевые слова для определения аварийных событий ML-моделью
EMERGENCY_KEYWORDS = [
    "преграда","препятствие","помеха","авария","сход","столкновение","пожар",
    "обрыв","излом","человек на путях","взрыв","взорвал","разрушен","не останавливается",
    "потерял управление","задымление","оползень","возгорание","горит","самоход",
    "проследовал запрещающий","машина на путях","застрял на переезде", "не тормозит", 
    "не реагирует на сигналы", "движется с превышением скорости",
]
FAULT_KEYWORDS = [
    "стрелк","светофор","неисправ","заклинило","задержк","буксов","колёсн",
    "не работает","сбой","отказ","не переводится","блокировка","нагрев",
    "ктсм","укспс","алсн","клуб","задержан",
]
CONSULT_KEYWORDS = [
    "что делать","как действ","как поступ","инструкция","регламент","объясни",
    "расскажи","помоги мне","почему","означает","правило","норматив","порядок действий",
    "действия при","расшифруй","поясни","можно ли","что значит","как обеспечить",
    "как восстанов","как устран", "интернет-поиск", "найди в интернете", "найди в сети",
    "сеть", "интернет", "гугл", "найди в гугле",
]
STATS_KEYWORDS = [
    "статистик","сколько","итого","всего записей","покажи журнал",
    "покажи статистик","за смену",
]
EVENT_KEYWORDS = [
    "поезд прибыл","поезд отправил","поезд стоит","поезд сошёл",
    "поезд застрял","поезд проследовал","поезд обнаружил",
    "прибытие поезда","отправление поезда","электричка","локомотив",
    "стрелка","светофор","сход вагона","заклинило","неисправность пути",
    "излом рельса","обрыв провода","путь перекрыт","преграда на пути",
    "препятствие на рельсах","пожар в вагоне","человек на путях",
    "столкновение поезд","грузовой поезд","пассажирский поезд",
    "прибыл на","отправился со","отправился с ","проследовал",
    "стоит на пути","стоит у запрещающего","обнаружил преграду",
    "застрял на переезде",
]

# ── Парсинг Final Answer из ответа агента ────────────────────────────────────
def _parse_agent_response(raw: str) -> str:
    """Убирает chain-of-thought, возвращает только Final Answer."""
    if not raw:
        return ""
    
    # Случай 1: агент вернул {"action": "Final Answer", "action_input": "текст"}
    fa_input = re.search(
        r'"action"\s*:\s*"Final Answer".*?"action_input"\s*:\s*"(.*?)"(?:\s*\})',
        raw, re.DOTALL
    )
    if fa_input:
        answer = fa_input.group(1).strip()
        return answer.replace("\\n", "\n").replace('\\"', '"')
    
    # Случай 2: Final Answer: текст
    fa_text = re.search(r'Final Answer[:\s]+(.+)', raw, re.DOTALL | re.IGNORECASE)
    if fa_text:
        return fa_text.group(1).strip()

    # Случай 3: агент вернул просто JSON action — значит застрял, берём Observation
    obs = re.findall(r'Observation:\s*(.+?)(?=\nThought:|\nAction:|\Z)', raw, re.DOTALL)
    if obs:
        return obs[-1].strip()

    # Fallback — убираем служебные строки
    lines = raw.split("\n")
    skip = ("Thought:", "Action:", "Observation:", "{", "}", "  \"", "```")
    clean = [l for l in lines if l.strip() and not any(l.strip().startswith(p) for p in skip)]
    return "\n".join(clean).strip() or raw


def classify_with_keywords(text: str, ml_class: str, ml_confidence: float) -> tuple:
    """
    Корректирует ML-классификацию на основе ключевых слов.
    Возвращает (class_name, confidence, overridden).
    """
    text_lower = text.lower()

    # Критические слова всегда переопределяют ML
    if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
        return "АВАРИЯ", 0.95, True

    # Если ML не уверен — помогаем ключевыми словами
    if ml_confidence < 0.65:
        if any(kw in text_lower for kw in FAULT_KEYWORDS):
            return "Тех. сбой", 0.70, True
        if ml_class == "Штатная" and ml_confidence < 0.60:
            return "Тех. сбой", 0.55, True

    return ml_class, ml_confidence, False


def process_event(text: str, source: str = "текст") -> dict:
    """Полный пайплайн: текст → ML → парсинг → RAG → инструкции → журнал."""
    result = {"text": text, "source": source}

    # 1. ML классификация + keyword correction (Требование 10)
    if st.session_state.ml_model:
        from ml_model.train_classifier import predict
        ml_result = predict(text, st.session_state.ml_model)
        # ИСПРАВЛЕНИЕ: classify_with_keywords теперь вызывается здесь
        final_class, final_conf, overridden = classify_with_keywords(
            text, ml_result["class_name"], ml_result["confidence"]
        )
        result["ml"] = ml_result
        result["ml_raw_class"] = ml_result["class_name"]
        result["ml_overridden"] = overridden
        result["event_class"] = final_class
        result["confidence"] = final_conf
    else:
        result["event_class"] = "Неизвестно"
        result["confidence"] = 0.0
        result["ml_overridden"] = False

    # 2. Парсинг сущностей (Цепочка 1, Требование 7)
    if st.session_state.entity_chain:
        result["parsed"] = st.session_state.entity_chain.parse(text)
    else:
        result["parsed"] = {"train_id": None, "track": None, "status": text[:50], "issue": None}

    # 3. RAG поиск (Требование 8)
    rag = st.session_state.rag_state
    rag_context = ""
    if rag["chunks"] and result["event_class"] in ("Тех. сбой", "АВАРИЯ"):
        from rag_data.build_knowledge_base import search
        rag_results = search(text, rag["index"], rag["chunks"], rag["model"], top_k=2)
        if rag_results:
            rag_context = "\n\n".join(r["content"] for r in rag_results)
    result["rag_context"] = rag_context

    # 4. Генерация инструкций (Цепочка 2, Требование 7)
    if st.session_state.instruction_chain and result["event_class"] != "Штатная":
        parsed = result.get("parsed", {})
        situation = (
            f"Поезд: {parsed.get('train_id', 'н/д')}, "
            f"Путь: {parsed.get('track', 'н/д')}, "
            f"Проблема: {parsed.get('issue') or text}"
        )
        result["instructions"] = st.session_state.instruction_chain.generate(
            situation=situation,
            event_class=result["event_class"],
            rag_context=rag_context,
        )
    else:
        result["instructions"] = None

    # 5. Запись в журнал — ВСЕГДА через прямую запись (Агент 1 опционально)
    # ИСПРАВЛЕНИЕ: confidence передаётся как float 0-1, а не как строка
    parsed = result.get("parsed", {})
    _direct_journal_write(parsed, result, source)

    _reload_journal()
    return result


def _direct_journal_write(parsed: dict, result: dict, source: str):
    """Запись в журнал без агента — надёжно и без галлюцинаций."""
    conf_val = result.get("confidence", 0.0)
    # Защита: если вдруг пришло > 1 (например 89.0 вместо 0.89)
    if conf_val > 1.0:
        conf_val = conf_val / 100.0
    new_row = {
        "Время": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "Поезд": parsed.get("train_id") or "—",
        "Путь":  parsed.get("track")    or "—",
        "Статус": parsed.get("status")  or result["text"][:50],
        "Класс события": result["event_class"],
        "Уверенность ML": f"{conf_val:.0%}",
        "Примечание": parsed.get("issue") or "",
        "Источник": source,  # передаётся как есть, без кавычек
    }
    st.session_state.journal_df = pd.concat(
        [st.session_state.journal_df, pd.DataFrame([new_row])],
        ignore_index=True,
    )
    from agents.langchain_agents import JOURNAL_PATH, ensure_journal
    ensure_journal()
    st.session_state.journal_df.to_csv(JOURNAL_PATH, index=False, encoding="utf-8-sig")


def _reload_journal():
    try:
        from agents.langchain_agents import JOURNAL_PATH
        if os.path.exists(JOURNAL_PATH):
            st.session_state.journal_df = pd.read_csv(JOURNAL_PATH, encoding="utf-8-sig")
    except Exception:
        pass


def get_real_stats() -> str:
    """Статистика прямо из CSV — без агента, без галлюцинаций."""
    _reload_journal()
    df = st.session_state.journal_df
    if df.empty:
        return "Журнал пуст — записей нет."
    total = len(df)
    lines = [f"**Статистика журнала за смену:** {total} записей"]
    for cls, icon in [("Штатная","🟢"),("Тех. сбой","🟡"),("АВАРИЯ","🔴")]:
        cnt = int((df["Класс события"] == cls).sum())
        if cnt:
            lines.append(f"{icon} {cls}: {cnt}")
    if "Источник" in df.columns:
        by_src = df["Источник"].value_counts().to_dict()
        src_str = ", ".join(f"{k}: {v}" for k, v in by_src.items())
        lines.append(f"Источники: {src_str}")
    return "\n".join(lines)


def consult_agent(query: str) -> str:
    """
    Агент 2 — Консультант (Требование 9).
    ConsultantAgent вызывает инструменты вручную (обход бага GigaChat).
    """
    if st.session_state.consultant_agent:
        try:
            # ConsultantAgent.run() — ручной вызов инструментов
            return st.session_state.consultant_agent.run(query)
        except Exception as e:
            if st.session_state.chat_chain:
                return st.session_state.chat_chain.chat(query)
            return f"Ошибка агента-консультанта: {e}"
 
    if st.session_state.chat_chain:
        return st.session_state.chat_chain.chat(query)
 
    # Fallback без LLM — только RAG
    rag = st.session_state.rag_state
    if rag.get("chunks"):
        hits = search(query, rag.get("index"), rag["chunks"], rag.get("model"), top_k=3)
        if hits:
            out = "**Найдено в нормативной базе РЖД:**\n\n"
            for i, r in enumerate(hits, 1):
                sec = r.get("metadata",{}).get("section","—")
                out += f"**[{i}] {sec}**\n{r['content'][:400]}\n\n"
            return out
    return "Введите GigaChat Credentials для активации LLM-функций."
# def consult_agent(query: str) -> str:
#     """Агент 2 — консультант. Парсит только Final Answer."""
#     if st.session_state.consultant_agent:
#         try:
#             raw = st.session_state.consultant_agent.run(query)
#             return _parse_agent_response(raw)
#         except Exception as e:
#             if st.session_state.chat_chain:
#                 return st.session_state.chat_chain.chat(query)
#             return f"Ошибка агента: {e}"
#     elif st.session_state.chat_chain:
#         return st.session_state.chat_chain.chat(query)

#     # Fallback: RAG без LLM
#     rag = st.session_state.rag_state
#     if rag.get("chunks"):
#         hits = search(query, rag.get("index"), rag["chunks"], rag.get("model"), top_k=3)
#         if hits:
#             out = "**Найдено в нормативной базе РЖД:**\n\n"
#             for i, r in enumerate(hits, 1):
#                 sec = r.get("metadata",{}).get("section","—")
#                 out += f"**[{i}] {sec}**\n{r['content'][:400]}\n\n"
#             return out
#     return "Введите GigaChat Credentials для активации LLM-функций."


def init_llm_components(api_key: str):
    """Инициализирует LLM, цепочки и агентов."""
    model_name = os.getenv("LLM_MODEL", "GigaChat")
    llm = GigaChat(
        credentials=api_key,
        verify_ssl_certs=False,
        model=model_name,
        temperature=0.2,
    )
    st.session_state.llm = llm
    st.session_state.entity_chain = EntityParserChain(llm)
    st.session_state.instruction_chain = InstructionGeneratorChain(llm)
    st.session_state.conv_manager = ConversationManager(max_messages=10)
    st.session_state.chat_chain = ChatChain(llm, st.session_state.conv_manager)
    rag = st.session_state.rag_state
    st.session_state.registrar_agent = create_registrar_agent(llm)
    st.session_state.consultant_agent = create_consultant_agent(llm, rag)
    st.session_state.api_key_set = True


@st.cache_resource(show_spinner="Загружаем ML-классификатор...")
def load_ml_model():
    from ml_model.train_classifier import load_model
    return load_model()


@st.cache_resource(show_spinner="Строим RAG-базу знаний...")
def load_rag():
    from rag_data.build_knowledge_base import load_or_build
    index, chunks, model = load_or_build()
    return {"index": index, "chunks": chunks, "model": model}


def _is_event(text: str) -> bool:
    """True только если текст — конкретное сообщение о движении, не вопрос."""
    t = text.lower().strip()
    # Вопросы никогда не являются событиями
    if t.endswith("?") or any(t.startswith(w) for w in
            ("что ", "как ", "почему ", "зачем ", "когда ", "где ", "кто ",
             "расскажи", "объясни", "какой", "какая", "какое", "можно")):
        return False
    return any(p in t for p in EVENT_KEYWORDS)


def clean_for_tts(text: str) -> str:
    """Убирает markdown, эмодзи и спецсимволы для синтеза речи."""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'__', '', text)
    cleaned = ''
    for ch in text:
        cat = unicodedata.category(ch)
        if cat not in ('So', 'Cn') and ord(ch) < 0x10000:
            cleaned += ch
        else:
            cleaned += ' '
    return re.sub(r' +', ' ', cleaned).strip()[:500]


def _make_tts(text: str) -> bytes:
    """Генерирует TTS только если включён флаг tts_enabled в session_state."""
    if not st.session_state.get("tts_enabled", False):
        return b""
    try:
        return text_to_speech(clean_for_tts(text)) or b""
    except Exception:
        return b""


def handle_user_input(user_input: str):
    """Единая точка обработки текстового ввода в чат."""
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    t = user_input.lower().strip()

    is_stats   = any(kw in t for kw in STATS_KEYWORDS)
    is_consult = (not is_stats) and any(kw in t for kw in CONSULT_KEYWORDS)
    is_event   = (not is_stats and not is_consult) and _is_event(user_input)

    if is_stats:
        response = get_real_stats()
        meta = {}

    elif is_consult:
        # Вопрос → консультант (не пишем в журнал)
        response = consult_agent(user_input)
        meta = {}

    elif is_event:
        # Событие → пайплайн → журнал
        event_result = process_event(user_input, source="чат")
        st.session_state.last_event = event_result
        cls  = event_result.get("event_class", "Неизвестно")
        conf = event_result.get("confidence", 0)
        parsed = event_result.get("parsed", {})

        parts = ["**Событие зафиксировано в журнале.**"]
        if parsed.get("train_id"):
            parts.append(f"🚂 Поезд **{parsed['train_id']}**, путь **{parsed.get('track','—')}**")
        if event_result.get("ml_overridden"):
            raw = event_result.get("ml_raw_class","")
            parts.append(f"*⚠️ ML предложил «{raw}», скорректировано → «{cls}»*")
        instr = event_result.get("instructions")
        if instr:
            parts.append(f"\n**Рекомендации:**\n{instr}")
        elif cls == "Штатная":
            parts.append("✅ Штатное событие. Дополнительных действий не требуется.")
        response = "\n\n".join(parts)
        meta = {"event_class": cls, "confidence": conf}

    else:
        response = consult_agent(user_input)
        meta = {}


    tts_bytes = _make_tts(response)
    msg = {"role": "assistant", "content": response, "meta": meta}
    if tts_bytes:
        msg["tts"] = tts_bytes
    st.session_state.chat_history.append(msg)

    # st.session_state.chat_history.append({
    #     "role": "assistant", "content": response, "meta": meta
    # })
