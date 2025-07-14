import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Configuration
st.set_page_config(page_title="Analyse de sentiment", page_icon="💬")
st.title("💬 Analyse de sentiment (XLM-Roberta)")
st.write("Entrez un avis en français pour déterminer s’il est **positif**, **négatif** ou **neutre**.")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Zone de saisie
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
                st.write(f"**Sentiment détecté** : {label}")
                st.progress(score / 100)
                st.write(f"**Confiance du modèle** : {score}%")
        else:
            st.warning("Le modèle n’a pas pu être chargé.")
    else:
        st.warning("Veuillez entrer un texte.")

