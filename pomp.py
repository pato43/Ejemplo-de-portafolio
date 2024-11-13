import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_iris
import numpy as np

# Configuración inicial de la página
st.set_page_config(page_title="Portafolio de Ciencia de Datos - Alexander Eduardo Rojas Garay", layout="wide")

# Título del Portafolio
st.title("Ejemplo de Portafolio de Ciencia de Datos e Información de Curso")
st.markdown("""
### Alexander Eduardo Rojas Garay
Bienvenido a un portafolio de ejemplo de ciencia de datos. En cada pestaña, exploro un modelo de Machine Learning con un conjunto de datos de muestra, explicando su contexto, funcionalidad y aplicación.
""")

# Información del curso
st.sidebar.title("Curso Basico de Python para Ciencia de Datos")
st.sidebar.markdown("""
**Duración:** 3 meses  
**Costo:** 150 por semana o 500 por el mes completo (1 semana GRATIS)  
**Certificación:** Constancia de Microsoft al finalizar  
[Registrarse aquí](https://forms.office.com/r/Mx65d2dHP9)  
[LinkedIn](https://www.linkedin.com/in/alexander-eduardo-rojas-garay-b17471235/)  
""")

# Temario del curso
st.sidebar.subheader("Temario del Curso")
with st.sidebar.expander("Módulo 1: Fundamentos de Python"):
    st.write("""
    - **Introducción a Python y Google Colab:** Configuración de Colab, primeras líneas de código y ventajas de Python en ciencia de datos.
    - **Estructuras Básicas de Python:** Variables, tipos de datos, estructuras de control (if, for, while) y funciones.
    - **Estructuras de Datos en Python:** Listas, tuplas, conjuntos y diccionarios: manipulación y casos de uso.
    """)

with st.sidebar.expander("Módulo 2: Manejo de Datos en Python"):
    st.write("""
    - **Numpy:** Arrays, operaciones aritméticas y estadísticas avanzadas.
    - **Pandas:** Creación y exploración de DataFrames y Series, limpieza y transformación de datos.
    """)

with st.sidebar.expander("Módulo 3: Estadística y Probabilidad para Ciencia de Datos"):
    st.write("""
    - **Estadística Descriptiva:** Medidas de tendencia central, dispersión, percentiles y visualización (boxplots, histogramas).
    - **Probabilidad:** Eventos, Teorema de Bayes, distribuciones (Binomial, Normal, Poisson, Exponencial).
    - **Inferencia Estadística:** Muestreo, intervalos de confianza, pruebas de hipótesis y aplicaciones en ciencia de datos.
    """)

with st.sidebar.expander("Módulo 4: Matemáticas para Ciencia de Datos"):
    st.write("""
    - **Álgebra Lineal:** Vectores, matrices, transformaciones lineales y descomposición en valores singulares (SVD).
    - **Cálculo y Optimización:** Derivadas, gradiente descendente, funciones de costo.
    - **Matemáticas Discretas:** Conjuntos, combinatoria y teoría de grafos.
    """)

with st.sidebar.expander("Módulo 5: Visualización de Datos"):
    st.write("""
    - **Matplotlib y Seaborn:** Gráficos básicos y avanzados (barras, líneas, heatmaps).
    - **Google Data Studio (Looker Studio):** Creación de reportes interactivos, alternativas a Power BI.
    """)

with st.sidebar.expander("Módulo 6: Introducción al Machine Learning"):
    st.write("""
    - **Fundamentos y Preparación de Datos**
    - **Modelos de Clasificación y Regresión Básica**
    - **Clustering y Reducción de Dimensionalidad**
    """)

with st.sidebar.expander("Módulo 7: Bases de Datos para Ciencia de Datos"):
    st.write("""
    - **SQL Básico y Conexión con Python**
    - **Consultas Avanzadas y Aplicaciones en Ciencia de Datos**
    """)

with st.sidebar.expander("Módulo 8: Proyecto Final de Ciencia de Datos"):
    st.write("""
    - **Definición de problema, modelado, análisis y presentación de resultados en Google Data Studio.**
    """)

# Cargar el dataset de ejemplo
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# División de los datos
X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Opciones de la barra lateral
st.sidebar.header("Modelos de Machine Learning")
page = st.sidebar.radio("Selecciona el modelo para ver detalles", 
                        ["Exploración de Datos", "Máquina de Vectores de Soporte (SVM)", "K-Nearest Neighbors (KNN)", 
                         "Clustering con KMeans", "Regresión Logística", "Árbol de Decisión"])

# Exploración de datos
if page == "Exploración de Datos":
    st.header("Exploración de Datos")
    st.write("Visualización general de los datos de muestra (dataset de flores Iris).")
    st.write(df.head())
    
    # Gráficos exploratorios
    st.subheader("Distribución de Clases")
    fig, ax = plt.subplots()
    sns.violinplot(x='target', data=df, inner='quartile')
    ax.set_xticklabels(data.target_names)
    st.pyplot(fig)
    
    st.subheader("Mapa de Calor de Correlaciones")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# SVM
elif page == "Máquina de Vectores de Soporte (SVM)":
    st.header("Máquina de Vectores de Soporte (SVM)")
    st.write("Este modelo clasifica los datos de Iris en tipos de flores basándose en el hiperplano óptimo.")
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    # Gráfico del hiperplano
    st.write("Visualización del Hiperplano:")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='viridis')
    ax.set_title("Separación de Clases SVM")
    st.pyplot(fig)

# Otras secciones de modelos (KNN, KMeans, Regresión Logística, Árbol de Decisión) se pueden agregar de manera similar...

# Footer
st.markdown("---")
st.markdown("Contacto: Alexander Eduardo Rojas Garay")
st.markdown("📞 Teléfono: 7225597963 | ✉️ Email: rojasalexander10@gmail.com")
