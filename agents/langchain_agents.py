"""
agents/langchain_agents.py
Два агента LangChain (Требование 9):
  Агент 1 «Регистратор»   — записывает события в journal.csv
  Агент 2 «Тех. консультант» — ищет по RAG + интернету
"""

import os
import json
import re
import pandas as pd

from datetime import datetime
from typing import Optional, List
from langchain_core.tools import Tool, BaseTool, StructuredTool
from langchain_classic.agents import AgentExecutor, AgentType, create_tool_calling_agent, initialize_agent
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rag_data.build_knowledge_base import search
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ── Путь к журналу ────────────────────────────────────────────────────────────

JOURNAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "journal.csv")
JOURNAL_COLUMNS = ["Время", "Поезд", "Путь", "Статус", "Класс события",
                   "Уверенность ML", "Примечание", "Источник"]


def ensure_journal():
    if not os.path.exists(JOURNAL_PATH):
        df = pd.DataFrame(columns=JOURNAL_COLUMNS)
        df.to_csv(JOURNAL_PATH, index=False, encoding="utf-8-sig")


# ── Инструменты Агента 1: Регистратор ────────────────────────────────────────

class JournalEntryInput(BaseModel):
    train_id: str = Field(description="Номер поезда")
    track: str = Field(description="Номер пути")
    status: str = Field(description="Статус события")
    event_class: str = Field(description="Класс: Штатная / Тех. сбой / АВАРИЯ")
    confidence: float = Field(description="Уверенность ML-классификатора (0-1)")
    note: Optional[str] = Field(default="", description="Дополнительное примечание")
    source: Optional[str] = Field(default="голос", description="Источник")


def write_journal_entry(
    train_id: str,
    track: str,
    status: str,
    event_class: str,
    confidence: float,
    note: str = "",
    source: str = "голос",
) -> str:
    ensure_journal()
    new_row = {
        "Время": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "Поезд": train_id or "—",
        "Путь": track or "—",
        "Статус": status,
        "Класс события": event_class,
        "Уверенность ML": f"{confidence:.0%}",
        "Примечание": note or "",
        "Источник": source,
    }
    df = pd.read_csv(JOURNAL_PATH, encoding="utf-8-sig")
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(JOURNAL_PATH, index=False, encoding="utf-8-sig")
    return f"✅ Запись добавлена: Поезд {train_id}, Путь {track}, {status} [{event_class}]"


def read_journal_last(n: str = "5") -> str:
    ensure_journal()
    df = pd.read_csv(JOURNAL_PATH, encoding="utf-8-sig")
    if df.empty:
        return "Журнал пуст."
    last = df.tail(int(n))
    return last.to_string(index=False)


def get_journal_stats(_: str = "") -> str:
    ensure_journal()
    df = pd.read_csv(JOURNAL_PATH, encoding="utf-8-sig")
    if df.empty:
        return "Журнал пуст, статистика недоступна."
    total = len(df)
    by_class = df["Класс события"].value_counts().to_dict()
    stats = f"Всего записей: {total}\n"
    for cls, cnt in by_class.items():
        stats += f"  {cls}: {cnt}\n"
    return stats


def check_train_schedule(query: str) -> str:
    """Проверяет расписание поезда по номеру из CSV."""
    schedule_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "rag_data", "schedule.csv"
    )
    if not os.path.exists(schedule_path):
        return f"Файл расписания не найден. Создайте {schedule_path}"
    df = pd.read_csv(schedule_path, encoding="utf-8-sig")
    matches = df[df.apply(lambda r: query.lower() in str(r).lower(), axis=1)]
    if matches.empty:
        return f"Поезд '{query}' не найден в расписании."
    return matches.to_string(index=False)


def build_registrar_tools() -> List[Tool]:
    return [
        StructuredTool.from_function(
            func=write_journal_entry,
            name="write_journal_entry",
            description=(
                "Добавляет запись в журнал движения поездов. "
                "Используй когда нужно зафиксировать событие: прибытие, отправление, неисправность."
            ),
            args_schema=JournalEntryInput,
        ),
        Tool(
            name="read_journal",
            func=read_journal_last,
            description="Читает последние N записей из журнала движения. Аргумент — строка с числом.",
        ),
        Tool(
            name="journal_stats",
            func=get_journal_stats,
            description="Показывает статистику по записям журнала (итого, по классам событий).",
        ),
        Tool(
            name="check_schedule",
            func=check_train_schedule,
            description="Проверяет расписание поезда по его номеру. Аргумент — номер или название поезда.",
        ),
    ]


# ── Инструменты Агента 2: Технический консультант ────────────────────────────

def search_regulations(query: str, rag_state: dict = None) -> str:
    """Поиск по базе нормативных документов (RAG)."""
    if rag_state is None:
        return "RAG-база недоступна. Убедитесь, что база знаний инициализирована."
    try:
        results = search(
            query,
            rag_state.get("index"),
            rag_state.get("chunks", []),
            rag_state.get("model"),
            top_k=3,
        )
        if not results:
            return "По данному запросу в нормативной базе ничего не найдено."
        output = "Найдено в нормативной базе РЖД:\n\n"
        for i, r in enumerate(results, 1):
            section = r.get("metadata", {}).get("section", "Раздел неизвестен")
            output += f"[{i}] {section}\n{r['content'][:400]}...\n\n"
        return output
    except Exception as e:
        return f"Ошибка поиска в RAG: {e}"


def search_internet(query: str) -> str:
    """Поиск актуальной информации в интернете через DuckDuckGo."""
    try:
        ddg = DuckDuckGoSearchRun()
        result = ddg.run(f"{query}")
        return result[:800] if result else "Интернет-поиск не дал результатов."
    except Exception as e:
        print(f"[DEBUG] Internet search error: {type(e).__name__}: {e}")
        return f"Интернет-поиск недоступен: {e}"


def explain_signal(signal: str) -> str:
    """Объясняет значение сигнала светофора или сигнала остановки."""
    signal_map = {
        "красный": "СТОП. Движение запрещено. Поезд должен остановиться до светофора.",
        "жёлтый": "Движение разрешено, но с готовностью остановиться. Скорость не более 60 км/ч.",
        "зелёный": "Движение разрешено с установленной скоростью.",
        "белый лунный": "Разрешение на маневровое движение.",
        "синий": "Запрещение маневрового движения.",
        "ч2": "Светофор Ч2 — входной/выходной, проверьте его состояние по системе СЦБ.",
    }
    key = signal.lower().strip()
    for k, v in signal_map.items():
        if k in key:
            return f"Сигнал '{signal}': {v}"
    return f"Значение сигнала '{signal}' уточните в Инструкции по сигнализации РЖД."


def build_consultant_tools(rag_state: dict = None) -> List[Tool]:
    def _rag_search(query: str) -> str:
        return search_regulations(query, rag_state)

    return [
        Tool(
            name="search_regulations",
            func=_rag_search,
            description=(
                "Ищет информацию в нормативной базе данных РЖД (инструкции, регламенты). "
                "Используй для вопросов 'что делать при...', 'как действовать если...'."
            ),
        ),
        Tool(
            name="search_internet",
            func=search_internet,
            description=(
                "Поиск актуальной информации в интернете. "
                "Используй если в нормативной базе нет ответа или нужна свежая информация."
            ),
        ),
        Tool(
            name="explain_signal",
            func=explain_signal,
            description=(
                "Объясняет значение сигнала светофора. "
                "Аргумент — цвет или обозначение сигнала (красный, жёлтый, Ч2 и т.д.)."
            ),
        ),
        Tool(
            name="journal_stats",
            func=get_journal_stats,
            description="Показывает статистику из журнала для анализа ситуации на станции.",
        ),
    ]


# ── Фабрика агентов ───────────────────────────────────────────────────────────

def create_registrar_agent(llm):
    """Агент 1: Регистратор событий."""
    tools = build_registrar_tools()
    system_msg = SystemMessage(content=(
        "Ты — Агент-Регистратор на железнодорожной станции. "
        "Твоя задача: точно и своевременно вносить события в журнал движения поездов. "
        "Используй инструменты для записи и чтения журнала. "
        "Всегда подтверждай выполненные записи. "
        "Если данные неполные — запиши что есть, добавив '(данные уточняются)'."
    ))
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={"system_message": system_msg},
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
    )


def create_consultant_agent(llm, rag_state: dict = None):
    tools = build_consultant_tools(rag_state)
    return ConsultantAgent(llm, rag_state, tools)

def create_consultant_agent_legacy(llm, rag_state: dict = None):
    tools = build_consultant_tools(rag_state)
    system_msg = SystemMessage(content=(
        "Ты — Консультант дежурного по станции на железной дороге.\n"
        "Отвечаешь на ЛЮБЫЕ вопросы пользователя, используя инструменты.\n\n"
        "КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:\n"
        "1. После строки action_input — НЕМЕДЛЕННО ОСТАНОВИСЬ. Не пиши ничего.\n"
        "2. НИКОГДА не пиши слово Observation самостоятельно.\n"
        "3. НИКОГДА не придумывай результат поиска.\n"
        "4. Система сама вызовет инструмент и вернёт результат.\n"
        "5. Только после получения реального Observation пиши Final Answer.\n\n"
        "ВЫБОР ИНСТРУМЕНТА:\n"
        "- Вопросы про нормативы РЖД → search_regulations\n"
        "- Погода, новости, дата, общее → search_internet\n"
        "- Сигналы светофора → explain_signal\n"
        "- Статистика журнала → journal_stats\n\n"
        "Формат ответа строго JSON:\n"
        "{\"action\": \"имя_инструмента\", \"action_input\": \"запрос\"}"
    ))

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "system_message": system_msg
        },
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate",
    )


class ConsultantAgent:
    """
    Агент 2 — Консультант (Требование 9).
    Ручное оркестрирование инструментов:
      Шаг 1: LLM выбирает инструмент и возвращает JSON.
      Шаг 2: Python вызывает реальную Python-функцию.
      Шаг 3: LLM формирует ответ на основе реального результата.
    """
 
    TOOL_SELECTION_PROMPT = (
        "Ты — Консультант ДСП на железнодорожной станции.\n"
        "Отвечаешь на вопросы используя инструменты.\n\n"
        "Инструменты и когда их использовать:\n"
        "- search_regulations: ВСЕ вопросы про железную дорогу, РЖД, поезда, "
        "сигналы, стрелки, светофоры, переговоры, регламенты, инструкции, "
        "действия при авариях, ПТЭ, ЦД-790, буксы, колёсные пары, сход вагонов\n"
        "- search_internet: погода, новости, дата, курсы валют, "
        "всё что НЕ связано с железной дорогой\n"
        "- explain_signal: только значение конкретного сигнала светофора\n"
        "- journal_stats: статистика журнала смены\n\n"
        "ГЛАВНОЕ ПРАВИЛО: если вопрос хоть как-то связан с ж/д — используй search_regulations.\n"
        "ИСКЛЮЧЕНИЕ: вопрос про тебя самого → {\"action\": \"no_tool\", \"action_input\": \"\"}\n\n"
        "Ответь ТОЛЬКО JSON:\n"
        '{"action": "имя_инструмента", "action_input": "запрос на русском"}'
    )
 
    def __init__(self, llm, rag_state=None, tools=None):
        self.llm = llm
        self._tool_map = {}
        if tools:
            for t in tools:
                self._tool_map[t.name] = t.func
        else:
            def _rag(q): return search_regulations(q, rag_state)
            self._tool_map = {
                "search_regulations": _rag,
                "search_internet":    search_internet,
                "explain_signal":     explain_signal,
                "journal_stats":      get_journal_stats,
            }
 
    def _parse_tool_call(self, text: str):
        import re as _re
        m = _re.search(
            r'\{[^{}]*"action"\s*:\s*"([^"]+)"[^{}]*"action_input"\s*:\s*"([^"]*)"[^{}]*\}',
            text, _re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        m2 = _re.search(
            r'\{[^{}]*"action_input"\s*:\s*"([^"]*)"[^{}]*"action"\s*:\s*"([^"]+)"[^{}]*\}',
            text, _re.DOTALL)
        if m2:
            return m2.group(2).strip(), m2.group(1).strip()
        return None
 
    def run(self, question: str) -> str:
        # Шаг 1: LLM выбирает инструмент
        r1 = self.llm.invoke([
            SystemMessage(content=self.TOOL_SELECTION_PROMPT),
            HumanMessage(content=question),
        ])
        text1 = r1.content if hasattr(r1, "content") else str(r1)
        print(f"[Agent2] Tool selection: {text1[:200]}")
 
        # Шаг 2: вызываем реальную Python-функцию
        call = self._parse_tool_call(text1)
        if call:
            action, action_input = call
            fn = self._tool_map.get(action, search_internet)
            print(f"[Agent2] Calling {action}(\'{action_input}\')")
            tool_result = fn(action_input)
        else:
            print("[Agent2] No JSON found, falling back to search_internet")
            tool_result = search_internet(question)
 
        # Шаг 3: LLM формирует финальный ответ
        r2 = self.llm.invoke([
            SystemMessage(content="Ты — помощник ДСП. Отвечай по-русски на основе данных поиска."),
            HumanMessage(content=(
                f"Вопрос: {question}\n\n"
                f"Результат поиска:\n{tool_result}\n\n"
                f"Дай полный и понятный ответ."
            )),
        ])
        return r2.content if hasattr(r2, "content") else str(r2)