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
Bienvenido a mi portafolio de ciencia de datos. En cada pestaña, exploro un modelo de Machine Learning con un conjunto de datos de muestra, explicando su contexto, funcionalidad y aplicación.
""")

# Información del curso
st.sidebar.title("Curso Basico de Python para Ciencia de Datos")
st.sidebar.markdown("""
**Duración:** 3 meses  
**Costo:** 130 por semana o 400 por el mes completo (1 semana GRATIS) 
**Certificación:** Constancia de Microsoft al finalizar  
[Registrarse aquí](https://forms.office.com/r/Mx65d2dHP9)  
[LinkedIn](https://linkedin.com/in/alexandereduardo)  
""")

# Temario del curso
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
    KMeans es un algoritmo de clustering sin supervisión. Agrupa datos en "clusters" o grupos según su similitud. Es útil para segmentación y descubrimiento de patrones sin etiquetas. En este caso, usamos KMeans para identificar grupos en el dataset de Iris.
    """)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X)
    
    fig, ax = plt.subplots()
    sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='kmeans_cluster', data=df, palette="viridis", ax=ax)
    st.pyplot(fig)

# Regresión Logística
elif page == "Regresión Logística":
    st.header("Regresión Logística")
    st.write("""
    La regresión logística predice probabilidades de una clase, ideal para clasificación binaria. Aquí se aplica a la clasificación de tipos de flores en el dataset de Iris, demostrando su versatilidad en problemas de múltiples clases.
    """)
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    
    # Curva ROC
    st.write("Curva ROC para Regresión Logística:")
    fig, ax = plt.subplots()
    sns.lineplot([0,1], [0,1], linestyle='--', ax=ax)
    # Agrega aquí la lógica de ROC si deseas detallar más
    ax.set_title("Curva ROC")
    st.pyplot(fig)

# Árbol de Decisión
elif page == "Árbol de Decisión":
    st.header("Árbol de Decisión")
    st.write("""
    Los árboles de decisión son modelos interpretables que dividen el espacio de características en regiones distintas para clasificar o predecir. Son útiles en aplicaciones donde se requiere transparencia y fácil interpretación.
    """)

    # Entrenamiento del modelo de árbol de decisión
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)
    
    # Realizar predicciones con el modelo
    y_pred = tree_model.predict(X_test)
    
    # Visualización del Árbol de Decisión
    st.write("Diagrama del Árbol de Decisión:")
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(tree_model, filled=True, feature_names=data.feature_names, class_names=data.target_names, ax=ax)
    st.pyplot(fig)

    # Resultados de la Matriz de Confusión
    st.write("Matriz de Confusión para Árbol de Decisión:")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    st.pyplot(fig)
    
    # Informe de clasificación para el modelo de Árbol de Decisión
    st.write("Informe de clasificación para Árbol de Decisión:")
    st.text(classification_report(y_test, y_pred))


# Footer
st.markdown("---")
st.markdown("Contacto: Alexander Eduardo Rojas Garay")
st.markdown("📞 Teléfono: 7225597963 | ✉️ Email: rojasalexander10@gmail.com")
