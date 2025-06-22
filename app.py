import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Título de la app
st.title("📊 Analizador de Sentimientos")

# Cargar modelo y vectorizador
@st.cache_resource
def cargar_modelo():
    model = joblib.load('modelo_sentimientos.pkl')
    vectorizer = joblib.load('vectorizador_tfidf.pkl')
    return model, vectorizer

model, vectorizer = cargar_modelo()

# Entrada de texto
user_input = st.text_area("✍️ Escribe un texto para analizar su sentimiento:", 
                         "Me encanta este producto, es increíble...")

# Botón para predecir
if st.button("🔍 Predecir"):
    # Preprocesar y vectorizar
    texto_vectorizado = vectorizer.transform([user_input])
    
    # Predecir
    prediccion = model.predict(texto_vectorizado)
    probabilidades = model.decision_function(texto_vectorizado)
    
    # Mostrar resultado
    st.subheader("Resultado:")
    st.write(f"**Sentimiento predicho:** {prediccion[0]}")
    
    # Opcional: Mostrar probabilidades (para LinearSVC)
    st.write("**Score de confianza:**")
    st.progress(float(probabilidades[0] + 10) / 20)  # Ajuste visual para valores negativos/positivos

    # Explicación
    st.info("""
    🔍 **Interpretación:**
    - **Positivo**: Score > 0 (más cercano a 1 = más positivo).
    - **Negativo**: Score < 0 (más cercano a -1 = más negativo).
    """)

# Sección opcional: Mostrar ejemplos
st.sidebar.markdown("## Ejemplos rápidos")
ejemplos = [
    "Odio cuando esto pasa... 😠",
    "Estoy muy feliz con el resultado 😊",
    "No es tan malo, pero podría mejorar."
]

for ejemplo in ejemplos:
    if st.sidebar.button(ejemplo):
        user_input = ejemplo