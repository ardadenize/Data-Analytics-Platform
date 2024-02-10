import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, confusion_matrix



def siniflandirma_analizi(uploaded_file):
    st.write("Sınıflandırm Analizi işlemleri")
    df = pd.read_csv(uploaded_file)
    st.write("Seçilen veri seti:")
    st.dataframe(df)

    target_column = st.selectbox("Select the Target Column", df.columns)
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    st.write('Shape of dataset:', X.shape)
    st.write('Number of classes:', len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifiers = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    st.write("Classifier Results:")
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[clf_name] = acc
        st.write(f"{clf_name}: Accuracy = {acc}")
        correct_predictions = np.sum(y_pred == y_test)
        total_samples = len(y_test)
        st.write(f"Toplam doğru sınıflandırılan örnek sayısı: {correct_predictions}/{total_samples}")

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        st.write(f"{clf_name} Sınıflandırma Sonuçları:")
        st.write(f"F1 Puanı (F1 Score): {f1}")
        st.write(f"Hassasiyet (Precision): {precision}")
        st.write("Karışıklık Matrisi (Confusion Matrix):")
        st.write(confusion)
        st.write(" \n ")




def siniflandirma_analizi_sonuclar():
    tab1, tab2, tab3 = st.tabs(["CSV", "XLSX", "TXT"])

    with tab1:
        uploaded_file_csv = st.file_uploader("Lütfen bir veri seti seçin (CSV)", type=["csv"])
        if uploaded_file_csv is not None:
            siniflandirma_analizi(uploaded_file_csv)
        else:
            st.write('Lütfen veri yükleyin...')

    with tab2:
        uploaded_file_xlsx = st.file_uploader("Lütfen bir veri seti seçin (XLSX)", type=["xlsx"])
        if uploaded_file_xlsx is not None:
            siniflandirma_analizi(uploaded_file_xlsx)
        else:
            st.write('Lütfen veri yükleyin...')

    with tab3:
        uploaded_file_txt = st.file_uploader("Lütfen bir veri seti seçin (TXT)", type=["txt"])
        if uploaded_file_txt is not None:
            siniflandirma_analizi(uploaded_file_txt)
        else:
            st.write('Lütfen veri yükleyin...')

 