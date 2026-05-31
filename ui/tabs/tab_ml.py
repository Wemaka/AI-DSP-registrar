import streamlit as st
import pandas as pd

from ui.logic.logic import classify_with_keywords

# ══════════════════════════════════════════════════════════════════════════════
# Вкладка 6: ML-модель
# ══════════════════════════════════════════════════════════════════════════════

def render_ml_tab():
    st.subheader("🤖 ML-классификатор событий")
    st.markdown("""
        **Алгоритм:** TF-IDF (char_wb, ngrams 2–4, max_features=5000) + LogisticRegression (C=5, balanced)  
        **Классы:** `0` Штатная &nbsp;|&nbsp; `1` Тех. сбой &nbsp;|&nbsp; `2` АВАРИЯ  
        **Датасет:** 70 примеров, стратифицированный 80/20 split, CV F1-macro ≈ 0.85  
        **Корректировка:** ключевые слова переопределяют ML при уверенности < 65%
    """)

    col_ml1, col_ml2 = st.columns([1, 1])

    with col_ml1:
        st.markdown("#### 🧪 Тест классификатора")
        test_text = st.text_area(
            "Введите текст для классификации:",
            value="Поезд 3002 стоит на пятом пути, заклинило колёсную пару",
            height=100,
            key="ml_test_text"
        )
        if st.button("Классифицировать", type="primary"):
            if st.session_state.ml_model:
                from ml_model.train_classifier import predict, CLASS_NAMES
                result = predict(test_text, st.session_state.ml_model)

                final_cls, final_conf, overridden = classify_with_keywords(
                    test_text, result["class_name"], result["confidence"]
                )

                icon = {"Штатная": "🟢", "Тех. сбой": "🟡", "АВАРИЯ": "🔴"}.get(final_cls, "⚪")
                st.markdown(f"### {icon} {final_cls}")
                if overridden:
                    st.warning(f"ML предложил «{result['class_name']}» ({result['confidence']:.1%}), "
                               f"скорректировано по ключевым словам → «{final_cls}»")
                else:
                    st.metric("Уверенность", f"{final_conf:.1%}")
                st.markdown("**Распределение вероятностей:**")
                
                for class_name, prob in result["probabilities"].items():
                    bar_icon = {"Штатная": "🟢", "Тех. сбой": "🟡", "АВАРИЯ": "🔴"}.get(class_name, "⚪")
                    st.progress(prob, text=f"{bar_icon} {class_name}: {prob:.1%}")
            else:
                st.error("ML-модель не загружена")

    with col_ml2:
        st.markdown("#### 📊 Переобучение модели")
        st.info("Модель обучается автоматически при первом запуске.\nДатасет: `ml_model/train_classifier.py`")

        if st.button("🔄 Переобучить модель", width="stretch"):
            with st.spinner("Обучаем модель..."):
                try:
                    from ml_model.train_classifier import train_and_save, load_model
                    metrics = train_and_save()
                    st.session_state.ml_model = load_model()

                    st.success(f"✅ Модель обучена!")
                    st.metric("CV F1-macro", f"{metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")

                    report = metrics.get("test_report", {})
                    rows = []
                    for cls_name in ["Штатная", "Тех. сбой", "АВАРИЯ"]:
                        if cls_name in report:
                            m = report[cls_name]
                            rows.append({
                                "Класс": cls_name,
                                "Precision": f"{m['precision']:.2f}",
                                "Recall": f"{m['recall']:.2f}",
                                "F1": f"{m['f1-score']:.2f}",
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch")
                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")

        st.divider()
        st.markdown("#### Примеры из датасета")
        examples = [
            ("Поезд 45 прибыл","Штатная 🟢"),
            ("Стрелка 12 не переводится","Тех. сбой 🟡"),
            ("Излом рельса на перегоне","АВАРИЯ 🔴"),
            ("Впереди преграда на пути","АВАРИЯ 🔴 (ключ. слово)"),
        ]
        for text, label in examples:
            st.caption(f"`{text}` → **{label}**")
