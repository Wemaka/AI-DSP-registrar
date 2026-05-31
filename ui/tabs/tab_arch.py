import streamlit as st

def render_arch_tab():
    st.subheader("🏗️ Архитектура системы")
    st.markdown("Здесь показано **где в коде** реализовано каждое требование.")

    st.markdown("### Поток обработки одной команды ДСП")
    st.markdown("""
        ```
            Голос/Текст ДСП
                │
                ▼
            [Whisper STT]  ←──── utils/audio_utils.py  (Требование 4)
                │
                ▼
            [ML-классификатор]  ←── ml_model/train_classifier.py  (Требование 10)
            TF-IDF + LogReg       Классы: Штатная / Тех.сбой / АВАРИЯ
            + Keyword correction  Если уверенность < 65% → ключевые слова
                │
                ▼
            [Цепочка 1: Парсер сущностей]  ←── chains/langchain_chains.py  (Требование 7)
            LLM промпт → JSON              EntityParserChain
            {train_id, track, status, issue}
                │
                ▼
            [RAG-поиск]  ←── rag_data/build_knowledge_base.py  (Требование 8)
            FAISS + sentence-transformers   Инструкция по сигнализации РЖД
            Топ-2 релевантных чанка
                │
                ▼
            [Цепочка 2: Генератор инструкций]  ←── chains/langchain_chains.py  (Требование 7)
            Ситуация + RAG контекст → инструкции  InstructionGeneratorChain
                │
                ▼
            [Агент-Регистратор]  ←── agents/langchain_agents.py  (Требование 9)
            Инструменты: write_journal_entry, read_journal, check_schedule
            → journal.csv
                │
                ▼
            [gTTS TTS]  ←── utils/audio_utils.py  (Требование 4)
            Озвучивает ответ
            ```
        """)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Требование 6 — Контекстное окно")
        st.markdown("""
            <div class="arch-box">
            <div class="arch-title">ConversationBufferWindowMemory</div>
            Файл: <code>chains/langchain_chains.py</code><br>
            Класс: <code>ConversationManager</code><br><br>
            Хранит последние <b>10 пар сообщений</b> (k=10).<br>
            Каждый запрос к LLM включает историю диалога:<br>
            <code>ДСП: &lt;сообщение&gt;</code><br>
            <code>Ассистент: &lt;ответ&gt;</code><br>
            → LLM понимает контекст разговора
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### Требование 7 — Две цепочки")
        st.markdown("""
            <div class="arch-box">
            <div class="arch-title">Цепочка 1: EntityParserChain</div>
            Файл: <code>chains/langchain_chains.py</code><br>
            <code>промпт | LLM | StrOutputParser</code><br>
            Вход: <i>"Поезд 3002 стоит на 5 пути, заклинило"</i><br>
            Выход: <code>{"train_id":"3002","track":"5","issue":"заклинивание"}</code>
            </div>
            <div class="arch-box">
            <div class="arch-title">Цепочка 2: InstructionGeneratorChain</div>
            Файл: <code>chains/langchain_chains.py</code><br>
            <code>промпт | LLM | StrOutputParser</code><br>
            Вход: ситуация + класс + RAG-контекст<br>
            Выход: пронумерованные инструкции для ДСП
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("### Требование 8 — RAG")
        st.markdown("""
            <div class="arch-box">
            <div class="arch-title">FAISS + sentence-transformers</div>
            Файл: <code>rag_data/build_knowledge_base.py</code><br><br>
            <b>Документы:</b> Инструкция по сигнализации РЖД<br>
            (встроенная) + загружаемые PDF/TXT<br><br>
            <b>Векторизация:</b> paraphrase-multilingual-MiniLM-L12<br>
            <b>Индекс:</b> IndexFlatIP (cosine similarity)<br>
            <b>Fallback:</b> keyword overlap без FAISS<br><br>
            При событии класса «Тех. сбой» или «АВАРИЯ»<br>
            → top-2 релевантных чанка → Цепочка 2
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### Требование 9 — Два агента")
        st.markdown("""
            <div class="arch-box">
            <div class="arch-title">Агент 1: Регистратор</div>
            Файл: <code>agents/langchain_agents.py</code><br>
            Тип: <code>STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION</code><br>
            Инструменты:<br>
            &nbsp;• <code>write_journal_entry</code> — запись в CSV<br>
            &nbsp;• <code>read_journal</code> — чтение журнала<br>
            &nbsp;• <code>journal_stats</code> — статистика<br>
            &nbsp;• <code>check_schedule</code> — расписание
            </div>
            <div class="arch-box">
            <div class="arch-title">Агент 2: Технический консультант</div>
            Файл: <code>agents/langchain_agents.py</code><br>
            Тип: <code>STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION</code><br>
            Инструменты:<br>
            &nbsp;• <code>search_regulations</code> — поиск в RAG<br>
            &nbsp;• <code>search_internet</code> — DuckDuckGo<br>
            &nbsp;• <code>explain_signal</code> — расшифровка сигналов<br>
            &nbsp;• <code>journal_stats</code> — статистика
            </div>
        """, unsafe_allow_html=True)