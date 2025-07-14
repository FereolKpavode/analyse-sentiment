import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Configuration de la page
st.set_page_config(page_title="Analyse de sentiment", page_icon="💬")
st.title("💬 Analyse de sentiment des avis clients (CamemBERT)")
st.write("Entrez un avis en français pour déterminer s’il est **positif** ou **négatif**.")

# Chargement du modèle avec mise en cache
@st.cache_resource
def load_model():
    try:
        model_name = "tblard/tf-allocine"
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, from_tf=True)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Zone de saisie utilisateur
user_input = st.text_area("✍️ Votre avis ici :", height=150)

# Bouton d'analyse
if st.button("Analyser le sentiment"):
    if user_input.strip():
        sentiment_pipeline = load_model()
        if sentiment_pipeline:
            with st.spinner("🔍 Analyse en cours..."):
                result = sentiment_pipeline(user_input)[0]
                label = result["label"]
                score = round(result["score"] * 100, 2)

                st.markdown("### 📊 Résultat de l'analyse")
                if label.upper() == "POSITIVE":
                    st.success(f"✅ Sentiment détecté : Positif ({score}%)")
                else:
                    st.error(f"⚠️ Sentiment détecté : Négatif ({score}%)")

                st.progress(score / 100)
                st.write(f"**Confiance du modèle** : {score}%")
        else:
            st.warning("Le modèle n'a pas pu être chargé.")
    else:
        st.warning("Veuillez entrer un texte à analyser.")
