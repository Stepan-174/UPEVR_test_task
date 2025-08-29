import streamlit as st
from transformers import pipeline
import torch  # добавляем импорт torch

@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_model

sentiment_pipeline = load_models()

st.title("Агент оценки звонков")
st.write("Введите расшифровку телефонного разговора:")

transcript = st.text_area("Текст разговора", height=300)

if st.button("Оценить"):
    if transcript.strip():
        result = sentiment_pipeline(transcript[:1000], max_length=512)
        label = result[0]['label']
        score = result[0]['score']

        if label == "POSITIVE":
            tone = "Положительный"
            recommendation = "Всё хорошо. Продолжайте поддерживать дружелюбный тон."
        elif label == "NEGATIVE":
            tone = "Негативный"
            recommendation = "Проявляйте больше дружелюбия и терпения."
        else:
            tone = "Нейтральный"
            recommendation = "Проявляйте больше дружелюбия."

        st.subheader("Результат оценки:")
        st.write(f"**Тон общения:** {tone}   (В процентах: {round(score * 100, 2)}%)")
        st.subheader("Рекомендации:")
        st.write(recommendation)
    else:
        st.warning("Введите текст разговора.")