import streamlit as st
from rag_data.build_knowledge_base import search

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка 5: База знаний RAG
# ══════════════════════════════════════════════════════════════════════════════

def render_rag_tab():
    st.subheader("📚 База знаний (RAG)")

    rag = st.session_state.rag_state
    chunks = rag.get("chunks", [])

    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.metric("Чанков в базе", len(chunks))
        st.metric("Статус FAISS", "✅ Загружен" if rag.get("index") else "⚠️ Без индекса")

        added = st.session_state.rag_added_files
        if added:
            st.markdown("**Добавленные документы:**")
            for f in added:
                st.caption(f"📄 {f.split('_')[0]}")

        # Тест поиска
        st.markdown("#### 🔍 Тест поиска по базе")
        test_query = st.text_input("Запрос:", value="что делать при сходе вагона")
        top_k = st.slider("Топ результатов:", 1, 5, 3)

        if st.button("Искать", type="primary"):
            results = search(test_query, rag.get("index"), chunks, rag.get("model"), top_k=top_k)
            if results:
                for i, r in enumerate(results, 1):
                    section = r.get("metadata", {}).get("section", "—")
                    score = r.get("score", 0)
                    with st.expander(f"[{i}] {section} (score: {score:.3f})"):
                        st.text(r["content"])
            else:
                st.warning("Ничего не найдено")

    with col_r2:
        st.markdown("#### 📄 Содержимое базы")
        if chunks:
            sections = list({c.get("metadata", {}).get("section", "—") for c in chunks})
            for sec in sorted(sections):
                sec_chunks = [c for c in chunks if c.get("metadata", {}).get("section") == sec]
                with st.expander(f"{sec} ({len(sec_chunks)} чанков)"):
                    for c in sec_chunks[:2]:
                        st.caption(c["content"][:200] + "...")
        else:
            st.info("База пуста. Документы загружаются автоматически при запуске.")
