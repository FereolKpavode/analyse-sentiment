import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Analyse de sentiment", page_icon="💬")
st.title("💬 Analyse de sentiment (CamemBERT-compatible)")
st.write("Entrez un avis en français pour déterminer s’il est **positif**, **négatif** ou **neutre**.")

@st.cache_resource
def load_model():
    try:
        return pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

user_input = st.text_area("✍️ Votre avis ici :", height=150)

if st.button("Analyser le sentiment"):
    if user_input.strip():
        sentiment_pipeline = load_model()
        if sentiment_pipeline:
            with st.spinner("Analyse en cours..."):
                result = sentiment_pipeline(user_input)[0]
                label = result["label"]
                score = round(result["score"] * 100, 2)

                st.markdown("### 📊 Résultat de l'analyse")
                st.success(f"Sentiment : {label} ({score}%)")
                st.progress(score / 100)
        else:
            st.warning("Le modèle n’a pas pu être chargé.")
    else:
        st.warning("Veuillez entrer un texte.")
