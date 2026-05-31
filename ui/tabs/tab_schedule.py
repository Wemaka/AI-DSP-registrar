import streamlit as st
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка 4: Расписание
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent.parent.parent

def render_schedule_tab():
    st.subheader("📅 Расписание поездов")

    schedule_path = ROOT / "rag_data" / "schedule.csv"
    if schedule_path.exists():
        sched_df = pd.read_csv(schedule_path, encoding="utf-8-sig")

        # Поиск по расписанию
        search_q = st.text_input(
            "🔍 Поиск по расписанию:", 
            placeholder="номер поезда, маршрут...", 
            key="schedule_search"
        )
        if search_q:
            mask = sched_df.apply(lambda r: search_q.lower() in str(r).lower(), axis=1)
            display_df = sched_df[mask]
        else:
            display_df = sched_df

        st.dataframe(display_df, width="stretch", height=400)

        # Загрузка нового расписания
        with st.expander("📤 Загрузить новое расписание (CSV)"):
            new_sched = st.file_uploader("CSV файл расписания", type=["csv"], key="sched_upload")
            if new_sched:
                new_df = pd.read_csv(new_sched, encoding="utf-8-sig")
                st.dataframe(new_df.head())
                if st.button("Применить расписание"):
                    new_df.to_csv(schedule_path, index=False, encoding="utf-8-sig")
                    st.success("✅ Расписание обновлено")
                    st.rerun()
    else:
        st.warning(f"Файл расписания не найден: {schedule_path}")
