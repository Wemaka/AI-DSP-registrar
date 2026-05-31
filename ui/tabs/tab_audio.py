import streamlit as st
from ui.logic.logic import process_event
from utils.audio_utils import transcribe_audio
from utils.audio_utils import text_to_speech, get_audio_html_player
from utils.audio_utils import text_to_speech, get_audio_html_player

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка 2: Голосовой ввод
# ══════════════════════════════════════════════════════════════════════════════

def render_audio_tab():
    st.subheader("🎙️ Голосовое управление")
    st.info("Нажмите на микрофон, произнесите команду, затем нажмите **«Обработать»**.")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### Запись команды")
        audio_input = st.audio_input("Говорите команду ДСП...", key="audio_recorder")

        if audio_input is not None:
            st.audio(audio_input)

            if st.button(
                "🔄 Распознать и обработать", 
                type="primary", width='stretch',
                key="process_audio_button"
            ):
                with st.spinner("Распознаём речь (Whisper)..."):
                    audio_bytes = audio_input.read()
                    result = transcribe_audio(audio_bytes)

                if result["error"]:
                    st.error(f"Ошибка STT: {result['error']}")
                    st.info("💡 Установите Whisper: `pip install openai-whisper` + ffmpeg")
                    demo_text = st.text_input(
                        "Или введите команду вручную:", key="audio_demo_text_input")
                    if demo_text:
                        result = {"text": demo_text, "error": None}
                    else:
                        result = None

                if result and result.get("text"):
                    recognized_text = result["text"]
                    st.success(f"✅ Распознано: **{recognized_text}**")

                    with st.spinner("Классифицируем и записываем..."):
                        # ИСПРАВЛЕНИЕ: source без кавычек — просто строка "голос"
                        event_result = process_event(recognized_text, source="голос")
                        st.session_state.last_event = event_result

                    cls  = event_result.get("event_class", "—")
                    conf = event_result.get("confidence", 0)

                    if cls == "АВАРИЯ":
                        st.markdown(
                            '<div class="critical-banner">⛔ КРИТИЧЕСКАЯ СИТУАЦИЯ — ТРЕБУЕТСЯ НЕМЕДЛЕННОЕ РЕАГИРОВАНИЕ</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        icon = {"Штатная": "🟢", "Тех. сбой": "🟡"}.get(cls, "⚪")
                        st.info(f"{icon} Класс события: **{cls}** (уверенность: {conf:.0%})")

                    instr = event_result.get("instructions")
                    response_text = instr or "✅ Штатное событие. Запись в журнале выполнена."

                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"🎙️ [Голос] {recognized_text}",
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "meta": {"event_class": cls, "confidence": conf},
                    })

                    if st.session_state.tts_enabled:
                        short_text = f"Класс {cls}. {response_text[:400]}"
                        audio_resp = text_to_speech(short_text.replace("**", "").replace("*", ""))
                        if audio_resp:
                            st.markdown("#### 🔊 Голосовой ответ:")
                            st.markdown(get_audio_html_player(audio_resp), unsafe_allow_html=True)
                            st.session_state.chat_history[-1]["tts"] = audio_resp
                    st.rerun()

    with col_b:
        st.markdown("#### Синтез речи (TTS)")
        tts_text_input = st.text_area(
            "Текст для озвучивания:",
            value="Поезд 2014, разрешаю отправление со второго пути.",
            height=100,
            key="tts_text_input",
        )
        tts_lang = st.selectbox("Язык:", ["ru", "en"], index=0, key="tts_lang_select")

        if st.button(
            "🔊 Озвучить", 
            width='stretch',
            key="tts_synthesize_button"
        ):
            with st.spinner("Синтезируем речь..."):
                audio_bytes = text_to_speech(tts_text_input, tts_lang)
            if audio_bytes:
                st.markdown(get_audio_html_player(audio_bytes), unsafe_allow_html=True)
                st.download_button(
                    "💾 Сохранить MP3",
                    data=audio_bytes,
                    file_name="tts_output.mp3",
                    mime="audio/mp3",
                    key="tts_download_button"
                )
            else:
                st.error("TTS недоступен. Установите: `pip install gTTS`")

        st.divider()
        st.markdown("#### 📋 Тестовые фразы ДСП")
        test_phrases = [
            "Поезд 2014 прибыл на первый путь",
            "Заклинило стрелку номер 12, движение ограничено",
            "Сход вагона с рельсов, требуется восстановительный поезд",
            "Поезд 3002 стоит у запрещающего сигнала Ч2",
        ]
        for i, phrase in enumerate(test_phrases):
            if st.button(phrase, width='stretch', key=f"audio_demo_{i}"):
                # ИСПРАВЛЕНИЕ: source="демо" без кавычек
                event_result = process_event(phrase, source="демо")
                st.session_state.last_event = event_result
                cls  = event_result.get("event_class", "—")
                conf = event_result.get("confidence", 0)
                icon = {"Штатная": "🟢", "Тех. сбой": "🟡", "АВАРИЯ": "🔴"}.get(cls, "⚪")
                st.success(f"{icon} {cls} ({conf:.0%}) — записано в журнал")
                st.rerun()