import os
import streamlit as st
import pandas as pd
from datetime import datetime
from pypdf import PdfReader
from rag_data.build_knowledge_base import build_faiss_index
from ui.logic.logic import init_llm_components, load_ml_model, load_rag
from utils.excel_export import export_to_excel
from agents.langchain_agents import JOURNAL_PATH

# ══════════════════════════════════════════════════════════════════════════════
# UI: Сайдбар
# ══════════════════════════════════════════════════════════════════════════════

def _add_document_to_rag(uploaded_file):
    """
    Добавляет загруженный документ в RAG-базу знаний (Требование 8).
    Защита от дублирования: проверяет имя файла в rag_added_files.
    """
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if file_key in st.session_state.rag_added_files:
        st.info(f"Документ «{uploaded_file.name}» уже добавлен в базу.")
        return

    try:
        if uploaded_file.type == "application/pdf":
            try:
                reader = PdfReader(uploaded_file)
                text = "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            text = uploaded_file.read().decode("utf-8", errors="ignore")

        if not text.strip():
            st.warning("Документ пустой или не удалось извлечь текст.")
            return

        # Разбиваем на смысловые чанки по 600 символов с перекрытием 100
        chunk_size, overlap = 600, 100
        new_chunks = []
        i = 0
        part_num = 0
        while i < len(text):
            chunk_text = text[i:i + chunk_size].strip()
            if chunk_text and len(chunk_text) > 50:  # пропускаем слишком короткие
                part_num += 1
                new_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": uploaded_file.name,
                        "section": f"{uploaded_file.name} (часть {part_num})",
                        "chunk_id": len(st.session_state.rag_state["chunks"]) + len(new_chunks),
                    }
                })
            i += chunk_size - overlap

        if not new_chunks:
            st.warning("Не удалось создать чанки из документа.")
            return

        rag = st.session_state.rag_state
        all_chunks = rag["chunks"] + new_chunks
        new_idx, new_list, new_model = build_faiss_index(all_chunks, rag.get("model"))
        st.session_state.rag_state = {"index": new_idx, "chunks": new_list, "model": new_model}
        st.session_state.rag_added_files.add(file_key)
        st.success(f"✅ Добавлено {len(new_chunks)} чанков из «{uploaded_file.name}»")

    except Exception as e:
        st.error(f"Ошибка добавления документа: {e}")

def render_sidebar():
    st.title("🚂 ДСП-Регистратор")
    st.caption("Интеллектуальный ассистент дежурного по станции")
    st.divider()

    # API ключ
    st.subheader("⚙️ Настройки")
    # api_key = st.text_input(
    #     "GigaChat API Key",
    #     type="password",
    #     value=os.getenv("GIGACHAT_API_KEY", ""),
    #     help="Необходим для LangChain цепочек и агентов",
    #     key="sidebar_api_key",
    # )
    api_key = os.getenv("GIGACHAT_API_KEY", "")
    if api_key and not st.session_state.api_key_set:
        with st.spinner("Инициализация LLM..."):
            try:
                init_llm_components(api_key)
                st.success("✅ LLM подключён")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    # Загрузка ML модели
    if st.session_state.ml_model is None:
        with st.spinner("Загружаем ML..."):
            try:
                st.session_state.ml_model = load_ml_model()
                st.success("✅ ML-модель готова")
            except Exception as e:
                st.warning(f"ML не загружен: {e}")

    # Загрузка RAG
    if not st.session_state.rag_state["chunks"]:
        with st.spinner("Строим RAG..."):
            try:
                st.session_state.rag_state = load_rag()
                st.success(f"✅ RAG: {len(st.session_state.rag_state['chunks'])} чанков")
            except Exception as e:
                st.warning(f"RAG: {e}")

    st.divider()

    # TTS toggle
    st.session_state.tts_enabled = st.toggle("🔊 Озвучивать ответы (TTS)", value=True)

    # Загрузка PDF документов в RAG
    st.subheader("📄 Загрузка документов в RAG")
    uploaded_pdf = st.file_uploader(
        "PDF/TXT инструкция",
        type=["pdf", "txt"],
        help="Загрузите нормативный документ для добавления в базу знаний"
    )
    if uploaded_pdf:
        with st.spinner("Добавляем в RAG..."):
            _add_document_to_rag(uploaded_pdf)

    st.divider()

    # Метрики сайдбара
    st.subheader("📊 Статистика смены")
    df = st.session_state.journal_df
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Записей", len(df))
    with col2:
        avariy = len(df[df["Класс события"] == "АВАРИЯ"]) if not df.empty else 0
        st.metric("⚠️ Аварий", avariy, delta_color="inverse")

    if not df.empty:
        for cls in ["Штатная", "Тех. сбой", "АВАРИЯ"]:
            cnt = len(df[df["Класс события"] == cls])
            color = {"Штатная": "🟢", "Тех. сбой": "🟡", "АВАРИЯ": "🔴"}.get(cls, "⚪")
            st.caption(f"{color} {cls}: {cnt}")

    st.divider()

    # Экспорт журнала
    st.subheader("💾 Экспорт")
    if not df.empty:
        excel_bytes = export_to_excel(df)
        st.download_button(
            "📥 Скачать Excel",
            data=excel_bytes,
            file_name=f"journal_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📄 Скачать CSV",
            data=csv_bytes,
            file_name="journal.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.caption("Журнал пуст")

    if st.button("🗑️ Очистить журнал", width="stretch"):
        st.session_state.journal_df = pd.DataFrame(columns=[
            "Время", "Поезд", "Путь", "Статус", "Класс события",
            "Уверенность ML", "Примечание", "Источник"
        ])
        if os.path.exists(JOURNAL_PATH):
            os.remove(JOURNAL_PATH)
        st.rerun()
