import streamlit as st

from ui.sidebar import render_sidebar
from ui.tabs.tab_arch import render_arch_tab
from ui.tabs.tab_audio import render_audio_tab
from ui.tabs.tab_chat import render_chat_tab
from ui.tabs.tab_journal import render_journal_tab
from ui.tabs.tab_ml import render_ml_tab
from ui.tabs.tab_rag import render_rag_tab
from ui.tabs.tab_schedule import render_schedule_tab

def render_all_ui():
    st.set_page_config(
        page_title="ДСП-Регистратор",
        page_icon="🚂",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .event-normal  { background:#1a3a1a; border-left:4px solid #4caf50; padding:10px; border-radius:6px; margin:4px 0; }
    .event-warning { background:#3a3a1a; border-left:4px solid #ffc107; padding:10px; border-radius:6px; margin:4px 0; }
    .event-danger  { background:#3a1a1a; border-left:4px solid #f44336; padding:10px; border-radius:6px; margin:4px 0; }
    .critical-banner {
        background: linear-gradient(90deg, #b71c1c, #f44336);
        color: white; font-size: 1.4em; font-weight: bold;
        text-align: center; padding: 18px; border-radius: 8px;
        animation: pulse 1s ease-in-out infinite alternate;
    }
    @keyframes pulse { from {opacity:1} to {opacity:0.75} }
    .arch-box { background:#1a1f2e; border:1px solid #2d3561; border-radius:8px; padding:14px; margin:6px 0; font-size:0.88em }
    .arch-title { color:#7eb8f7; font-weight:bold; margin-bottom:6px }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        render_sidebar()

    st.title("🚂 Умный ДСП-Регистратор")
    st.caption("AI-ассистент дежурного по станции | GigaChat + LangChain + RAG + ML")

    tab_chat, tab_audio, tab_journal, tab_rag, tab_ml, tab_arch = st.tabs([
        "💬 Регистрация", "🎙️ Голос",
        "📋 Журнал", "📚 База знаний", "🤖 ML-модель", "🏗️ Архитектура"
    ])

    with tab_chat: render_chat_tab()

    with tab_audio: render_audio_tab()

    with tab_journal: render_journal_tab()

    with tab_rag: render_rag_tab()

    with tab_ml: render_ml_tab()

    with tab_arch: render_arch_tab()


