# 🌟 Analizador de Sentimientos para Redes Sociales

[![Streamlit](https://img.shields.io/badge/Deployed_with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E?logo=scikit-learn)](https://scikit-learn.org/)

## 📌 Visión General
Sistema de NLP que **clasifica automáticamente emociones en textos** de redes sociales con:
- **89% de precisión** en datos reales
- Interfaz intuitiva con visualización interactiva
- Pipeline completo de ML (desde preprocesamiento hasta despliegue)


## 🛠️ Tecnologías Clave
```python
# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer  # Vectorización
from sklearn.svm import LinearSVC  # Modelo de clasificación

# Despliegue
import streamlit as st  # Interfaz web
import joblib  # Serialización
```

## 📦 Dependencias y Estructura

### 🔧 Versiones Clave
| Librería       | Versión   | Uso Principal                |
|----------------|-----------|------------------------------|
| `streamlit`    | `1.35.0`  | Interfaz web interactiva     |
| `scikit-learn` | `1.4.0`   | Modelado ML (SVM + TF-IDF)   |
| `pandas`       | `2.2.1`   | Procesamiento de datos       |
| `joblib`       | `1.3.2`   | Serialización del modelo     |

### 🌳 Estructura del Proyecto
```bash
.
├── 📜 app.py                  # Aplicación principal (Streamlit)
├── 🗃️ modelo_sentimientos.pkl  # Modelo SVM entrenado
├── 🔠 vectorizador_tfidf.pkl   # Vectorizador guardado
├── 📋 requirements.txt        # Entorno reproducible
└── 📖 README.md               # Documentación completa
