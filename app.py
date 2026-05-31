"""
app.py — «Умный ДСП-Регистратор»
Мультимодальный чат-бот для автоматизации ведения журнала движения поездов.

1. Web-приложение Streamlit
2. Элементы управления (кнопки, вкладки, слайдеры, загрузка файлов)
3. Ввод и генерация текста (чат, LLM)
4. Аудио-модальность (Whisper STT + gTTS TTS)
5. Внешние данные (schedule.csv + PDF инструкции через RAG + DuckDuckGo)
6. Контекстное окно LangChain (ConversationBufferWindowMemory)
7. Две LangChain-цепочки (Парсер сущностей + Генератор инструкций)
8. RAG (FAISS + sentence-transformers + нормативные документы РЖД)
9. Два агента (Регистратор + Технический консультант)
10. ML-классификатор (TF-IDF + LogisticRegression, sklearn)
"""

import sys
import pandas as pd
import streamlit as st
import warnings

from pathlib import Path
from dotenv import load_dotenv
from ui.logic.logic import handle_user_input
from ui.render_ui import render_all_ui

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*transformers.*")

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# ══════════════════════════════════════════════════════════════════════════════
# Инициализация session_state
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "api_key_set": False,
        "llm": None,
        "ml_model": None,
        "rag_state": {"index": None, "chunks": [], "model": None},
        "registrar_agent": None,
        "consultant_agent": None,
        "entity_chain": None,
        "instruction_chain": None,
        "chat_chain": None,
        "conv_manager": None,
        # [{role, content, meta}]
        "chat_history": [
            {
                "role": "assistant",
                "content": (
                    "👋 Добро пожаловать! Я **ДСП-Регистратор** — AI-ассистент дежурного по станции.\n\n"
                    "**Что я умею:**\n"
                    "🚂 **Фиксировать события** — напишите как есть:\n"
                    "&nbsp;&nbsp;*«Поезд 2014 прибыл на первый путь»*\n"
                    "&nbsp;&nbsp;*«Заклинило стрелку номер 12»*\n"
                    "&nbsp;&nbsp;*«Сход вагона с рельсов, требуется помощь»*\n\n"
                    "📚 **Отвечать на вопросы** по нормативам РЖД:\n"
                    "&nbsp;&nbsp;*«Что делать при заклинивании колёсной пары?»*\n"
                    "&nbsp;&nbsp;*«Что означает красный сигнал светофора?»*\n\n"
                    "🌐 **Искать в интернете** если нет в нормативах:\n"
                    "&nbsp;&nbsp;*«Какая погода на станции Люблино?»*\n\n"
                    "📊 **Показать статистику смены** — напишите *«статистика»*\n\n"
                    "💡 Используйте **быстрые команды** справа для примеров."
                ),
                "meta": {},
            }
        ],
        "journal_df": pd.DataFrame(columns=[
            "Время", "Поезд", "Путь", "Статус", "Класс события",
            "Уверенность ML", "Примечание", "Источник"
        ]),
        "last_event": None,
        "tts_enabled": True,
        "pending_input": None, 
        "whisper_model": "base",
        "rag_added_files": set(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

if st.session_state.pending_input:
    inp = st.session_state.pending_input
    st.session_state.pending_input = None
    handle_user_input(inp)

render_all_ui()