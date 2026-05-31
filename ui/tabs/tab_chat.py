import base64
import streamlit as st
from ui.logic.logic import consult_agent, handle_user_input, process_event
from utils.audio_utils import get_audio_html_player

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка 1: Чат
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_tab():
    col_main, col_info = st.columns([2, 1])

    with col_main:
        chat_container = st.container(height=460)
        with chat_container:
            for msg in st.session_state.chat_history:
                role    = msg["role"]
                content = msg["content"]
                meta    = msg.get("meta", {})

                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(content)
                else:
                    with st.chat_message("assistant"):
                        if meta.get("event_class") == "АВАРИЯ":
                            st.markdown(
                                '<div class="critical-banner">⛔ КРИТИЧЕСКАЯ СИТУАЦИЯ — ТРЕБУЕТСЯ НЕМЕДЛЕННОЕ РЕАГИРОВАНИЕ</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown(content)
                        if meta.get("event_class"):
                            cls   = meta["event_class"]
                            conf  = meta.get("confidence", 0)
                            color = {"Штатная":"🟢","Тех. сбой":"🟡","АВАРИЯ":"🔴"}.get(cls,"⚪")
                            st.caption(f"{color} Класс: **{cls}** | ML: {conf:.0%}")

                        if msg.get("tts"):
                            audio_b64 = base64.b64encode(msg["tts"]).decode()
                            autoplay = "autoplay" if st.session_state.get("tts_enabled", False) else ""

                            st.markdown(
                                f'''
                                <audio {autoplay} controls>
                                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                                </audio>
                                ''', unsafe_allow_html=True
                            )

        user_input = st.chat_input("Введите сообщение или команду ДСП...")
        if user_input:
            with st.spinner("Обрабатываем..."):
                handle_user_input(user_input)
            st.rerun()

    with col_info:
        st.subheader("🔍 Последнее событие")
        last = st.session_state.last_event
        if last:
            cls  = last.get("event_class","—")
            conf = last.get("confidence", 0)
            parsed = last.get("parsed", {})
            css  = {"Штатная":"event-normal","Тех. сбой":"event-warning","АВАРИЯ":"event-danger"}.get(cls,"event-normal")
            icon = {"Штатная":"🟢","Тех. сбой":"🟡","АВАРИЯ":"🔴"}.get(cls,"⚪")
            overridden = last.get("ml_overridden", False)
            raw_cls    = last.get("ml_raw_class", cls)
            st.markdown(f"""
                <div class="{css}">
                <b>{icon} {cls}</b>{"  ⚠️ скорректировано" if overridden else ""}<br>
                Уверенность: {conf:.0%}{"  (было: "+raw_cls+")" if overridden else ""}<br>
                Поезд: {parsed.get('train_id') or '—'}&nbsp;&nbsp;Путь: {parsed.get('track') or '—'}
                Статус: {str(parsed.get('status','—'))[:45]}
                </div>
            """, unsafe_allow_html=True)
            if last.get("rag_context"):
                with st.expander("📚 Найдено в нормативах"):
                    st.caption(last["rag_context"][:400])
        else:
            st.info("Событий ещё нет")

        st.divider()
        st.subheader("⚡ Быстрые команды")
        quick_cmds = [
            ("🚂 Прибытие",     "Поезд 2014 прибыл на первый путь"),
            ("🚨 Авария",       "Сход вагона с рельсов, требуется помощь"),
            ("⚠️ Сбой стрелки", "Заклинило стрелку номер 12"),
            ("🔴 Светофор",     "Поезд 3002 стоит у запрещающего сигнала Ч2"),
            ("🚧 Преграда",     "Поезд 1234 обнаружил преграду на 4 пути"),
            ("❓ Инструкция",   "Что делать при заклинивании колёсной пары?"),
            ("📊 Статистика",   "Покажи статистику журнала за смену"),
        ]
        for i, (label, cmd) in enumerate(quick_cmds):
            if st.button(label, width='stretch', key=f"chat_qcmd_{i}"):
                st.session_state.pending_input = cmd
                st.rerun()

        st.divider()
        if st.button("🗑️ Очистить чат", width='stretch'):
            st.session_state.chat_history = []
            if st.session_state.conv_manager:
                st.session_state.conv_manager.clear()
            st.rerun()
