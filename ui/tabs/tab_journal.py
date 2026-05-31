import streamlit as st
from ui.logic.logic import process_event
from datetime import datetime
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка 3: Журнал движения
# ══════════════════════════════════════════════════════════════════════════════

def render_journal_tab():
    st.subheader("📋 Журнал движения поездов")
    df = st.session_state.journal_df

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_class = st.multiselect(
            "Фильтр по классу:",
            options=["Штатная", "Тех. сбой", "АВАРИЯ"],
            default=["Штатная", "Тех. сбой", "АВАРИЯ"],
        )
    with col_f2:
        filter_train = st.text_input("Фильтр по поезду:", placeholder="номер поезда")
    with col_f3:
        # ИСПРАВЛЕНИЕ: добавлен "ручной" в список источников
        filter_source = st.multiselect(
            "Источник:",
            options=["голос", "текст", "чат", "демо", "ручной"],
            default=["голос", "текст", "чат", "демо", "ручной"],
        )

    filtered_df = df.copy()
    if filter_class:
        filtered_df = filtered_df[filtered_df["Класс события"].isin(filter_class)]
    if filter_train:
        filtered_df = filtered_df[
            filtered_df["Поезд"].astype(str).str.contains(filter_train, case=False, na=False)
        ]
    if filter_source and not filtered_df.empty:
        filtered_df = filtered_df[filtered_df["Источник"].isin(filter_source)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Всего записей", len(filtered_df))
    m2.metric("🟢 Штатных",  int((filtered_df["Класс события"]=="Штатная").sum())  if not filtered_df.empty else 0)
    m3.metric("🟡 Сбоев",    int((filtered_df["Класс события"]=="Тех. сбой").sum()) if not filtered_df.empty else 0)
    m4.metric("🔴 Аварий",   int((filtered_df["Класс события"]=="АВАРИЯ").sum())    if not filtered_df.empty else 0)

    if not filtered_df.empty:
        # ИСПРАВЛЕНИЕ: цвета работают и на светлой и на тёмной теме
        # Используем явные hex-цвета с хорошим контрастом
        def color_rows(row):
            cls = row.get("Класс события", "")
            colors = {
                "Штатная":   "background-color:#d4edda; color:#155724",   # зелёный
                "Тех. сбой": "background-color:#fff3cd; color:#856404",   # жёлтый
                "АВАРИЯ":    "background-color:#f8d7da; color:#721c24",   # красный
            }
            style = colors.get(cls, "")
            return [style] * len(row)

        styled = filtered_df.style.apply(color_rows, axis=1)
        st.dataframe(styled, width='stretch', height=380)
    else:
        st.info("📭 Нет записей. Введите команду в чате или используйте голосовой ввод.")

    with st.expander("➕ Добавить запись вручную"):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            m_train = st.text_input("Поезд:", key="manual_train")
            m_track = st.text_input("Путь:", key="manual_track")
        with col_m2:
            m_status = st.text_input("Статус:", key="manual_status")
            m_class  = st.selectbox(
                "Класс:", ["Штатная","Тех. сбой","АВАРИЯ"], 
                key="manual_class")
        with col_m3:
            m_note = st.text_area(
                "Примечание:", 
                key="manual_note",
                height=80)

        if st.button("💾 Добавить запись", type="primary"):
            if m_status:
                new_row = {
                    "Время": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                    "Поезд": m_train or "—",
                    "Путь":  m_track or "—",
                    "Статус": m_status,
                    "Класс события": m_class,
                    "Уверенность ML": "ручной ввод",
                    "Примечание": m_note,
                    "Источник": "ручной",
                }
                st.session_state.journal_df = pd.concat(
                    [st.session_state.journal_df, pd.DataFrame([new_row])],
                    ignore_index=True,
                )
                from agents.langchain_agents import JOURNAL_PATH
                st.session_state.journal_df.to_csv(JOURNAL_PATH, index=False, encoding="utf-8-sig")
                st.success("✅ Запись добавлена")
                st.rerun()
