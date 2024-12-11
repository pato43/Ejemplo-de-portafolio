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
st.set_page_config(page_title="Ejemplo de portafolio y registro", layout="wide")

# Título del Portafolio
st.title("Ejemplo de Portafolio de Ciencia de Datos - Información del Curso (en la barra lateral)")

# Descripción inicial
st.markdown("""
### Bienvenido/a a este portafolio de ciencia de datos
Este es un portafolio de **ejemplo** que muestra algunos de los análisis y modelos que realizaremos en el curso de Python para Ciencia de Datos. Aquí puedes explorar diferentes técnicas de Machine Learning aplicadas al dataset de Iris, observar visualizaciones y revisar el temario del curso.

**Objetivo**: que los estudiantes comprendan las técnicas de análisis de datos y visualización para poder aplicarlas en sus propios proyectos.  
""")

# Información del curso
st.sidebar.title("Curso de Python para Ciencia de Datos")
st.sidebar.markdown("""
**Duración:** 3 meses  
**Costo:** 250 por semana o 800 por el mes completo   
**Certificación:** Constancia de Microsoft al finalizar  
""")

# Botón de registro en el curso
if st.sidebar.button("Registrarse en el Curso"):
    st.sidebar.write("[Haz clic aquí para registrarte](https://forms.office.com/r/Mx65d2dHP9)")

# Agregar enlaces adicionales en la barra lateral
st.sidebar.markdown("### Recursos Adicionales")
st.sidebar.markdown("[Portafolio Personal](https://portafolio-personal-v1-es.streamlit.app/)")
st.sidebar.markdown("[Proyectos en Desarrollo](https://ptpart.streamlit.app/)")
st.sidebar.markdown("[Modelos y Actividades del Curso](https://modelosygraficos.streamlit.app/)")
st.sidebar.markdown("[Test de Conocimientos Previos](https://evaluaciontestdeconocimientosprevios.streamlit.app/)")

st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/alexander-eduardo-rojas-garay-b17471235/)")

# Temario del curso detallado
st.sidebar.subheader("Temario del Curso")
with st.sidebar.expander("Módulo 1: Fundamentos de Python"):
    st.write("""
    - Introducción a Python y Google Colab  
    - Estructuras básicas de control y datos en Python  
    - Funciones y estructuras de datos avanzadas  
    """)

with st.sidebar.expander("Módulo 2: Manejo de Datos en Python"):
    st.write("""
    - Manipulación de arrays y operaciones avanzadas con Numpy  
    - Transformación y limpieza de datos con Pandas  
    """)

with st.sidebar.expander("Módulo 3: Estadística y Probabilidad"):
    st.write("""
    - Estadística descriptiva y visualización de datos  
    - Conceptos básicos de probabilidad y distribuciones  
    - Inferencia estadística y pruebas de hipótesis  
    """)

with st.sidebar.expander("Módulo 4: Matemáticas para Ciencia de Datos"):
    st.write("""
    - Álgebra lineal aplicada a ciencia de datos  
    - Cálculo para optimización y modelos  
    - Matemáticas discretas y teoría de grafos  
    """)

with st.sidebar.expander("Módulo 5: Visualización de Datos"):
    st.write("""
    - Gráficos avanzados con Matplotlib y Seaborn  
    - Creación de reportes interactivos en Google Data Studio (Looker Studio)  
    """)

with st.sidebar.expander("Módulo 6: Introducción al Machine Learning"):
    st.write("""
    - Preparación de datos para modelos de ML  
    - Modelos de clasificación y regresión básicos  
    - Clustering y reducción de dimensionalidad  
    """)

with st.sidebar.expander("Módulo 7: Bases de Datos para Ciencia de Datos"):
    st.write("""
    - Consultas SQL básicas y avanzadas  
    - Conexión y manejo de datos en Python  
    """)

with st.sidebar.expander("Módulo 8: Proyecto Final de Ciencia de Datos"):
    st.write("""
    - Definición del problema, modelado y análisis  
    - Presentación de resultados en Looker Studio  
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
page = st.sidebar.radio("Explora el portafolio de ejemplo", 
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
    st.write("""
    El modelo SVM es ampliamente utilizado para clasificación en casos con clases claramente separables. Este modelo busca encontrar un hiperplano óptimo que maximice la separación entre las clases. Aquí se utiliza para clasificar tipos de flores en el dataset de Iris.
    """)
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    # Gráfico del hiperplano
    st.write("Visualización del Hiperplano:")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='viridis')
    ax.set_title("Separación de Clases SVM")
    st.pyplot(fig)

# K-Nearest Neighbors (KNN)
elif page == "K-Nearest Neighbors (KNN)":
    st.header("K-Nearest Neighbors (KNN)")
    st.write("""
    El modelo KNN clasifica un nuevo punto basándose en la clase mayoritaria de sus puntos vecinos más cercanos. Es ideal para problemas donde las clases están bien definidas y funciona bien con conjuntos de datos pequeños. En este ejemplo, se usa KNN para clasificar tipos de flores en el dataset de Iris.
    """)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    
    # Gráfico de dispersión 3D
    st.write("Gráfico de Dispersión 3D de KNN:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], X_test.iloc[:, 2], c=y_pred, cmap='cool')
    ax.set_title("Clasificación con KNN")
    st.pyplot(fig)

# KMeans
elif page == "Clustering con KMeans":
    st.header("Clustering con KMeans")
    st.write("""
    KMeans es un método de agrupamiento que busca particionar los datos en K grupos, donde cada dato pertenece al grupo con el centroide más cercano. Este algoritmo es útil para encontrar patrones en datos no etiquetados. En este ejemplo, aplicamos KMeans al dataset de Iris para agrupar las flores en 3 clusters.
    """)
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(X)
    df['cluster'] = kmeans_model.labels_
    
    # Visualización de clusters
    st.write("Visualización de Clusters:")
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='cluster', data=df, palette='viridis')
    ax.set_title("Clusters generados por KMeans")
    st.pyplot(fig)

# Regresión Logística
elif page == "Regresión Logística":
    st.header("Regresión Logística")
    st.write("""
    La regresión logística es un modelo de clasificación que utiliza una función logística para predecir la probabilidad de pertenencia a una clase. En este ejemplo, se aplica para clasificar tipos de flores en el dataset de Iris.
    """)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    
    # Reporte de métricas
    st.write("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))
    
    # Gráfico de matriz de confusión
    st.write("Matriz de Confusión:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='coolwarm', fmt='d')
    st.pyplot(fig)

# Árbol de Decisión
elif page == "Árbol de Decisión":
    st.header("Árbol de Decisión")
    st.write("""
    Los árboles de decisión son modelos interpretables que dividen los datos en nodos basados en reglas simples. En este ejemplo, se utiliza un árbol de decisión para clasificar tipos de flores en el dataset de Iris.
    """)
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    
    # Gráfico de importancia de características
    st.write("Importancia de las Características:")
    feature_importances = pd.Series(tree_model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    feature_importances.sort_values().plot(kind='barh', ax=ax, color='skyblue')
    st.pyplot(fig)


# Footer con redes sociales
st.markdown("---")
st.markdown("Sigue a Alexander Eduardo Rojas Garay en [LinkedIn](https://www.linkedin.com/in/alexander-eduardo-rojas-garay-b17471235/).")
