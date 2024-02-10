import streamlit as st
import pandas as pd 
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from dateutil import parser
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt



def veri_analizi(uploaded_file):
        df = pd.read_csv(uploaded_file)
        for col in df.columns:
            try:
                # Tarih sütunu olup olmadığını belirlemek için bir örnek veri kullanın
                sample = df[col].dropna().iloc[0]
                parser.parse(sample)

                # Tarih sütunu ise, tarih veri tipine dönüştürün
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

            except (TypeError, ValueError):
                # Tarih sütunu değilse, bir sonraki sütuna geçin
                continue
        st.write("Seçilen veri seti:")
        st.dataframe(df)

        # Veri setinin analizini gerçekleştirmek için aşağıdaki kodu kullanabilirsiniz
        with st.spinner('Veri seti analiz ediliyor...'):

            report = ProfileReport(df)
            st_profile_report(report)
        
        # Aykırı değer analizi butonunu ekleyelim
        st.write("Aykırı Değer Analizi")
        st.write("Veri setindeki numerik değerlerin aykırı değerlerini gösteren bir grafik oluşturulacaktır.")
        # Kullanıcıdan değişken adını alalım
        column_name = st.text_input("Lütfen aykırı değer analizi yapmak istediğiniz değişkenin adını girin:")
        if column_name:
            # Değişken adı girildiyse, değişkenin aykırı değerlerini gösteren bir grafik oluşturalım
            if column_name in df.select_dtypes(include=[np.number]).columns:
                fig, ax = plt.subplots(figsize=(10,6))
                sns.boxplot(x=df[column_name], ax=ax)
                st.pyplot(fig)
            else:
                st.write(f"Hata: {column_name} değişkeni numerik bir değişken değil.")

    
            
def veri_analizi_sonuclar():
        tab1, tab2, tab3 = st.tabs(["CSV", "XLSX", "TXT"])

        """with tab1:
            uploaded_file_csv = st.file_uploader("Lütfen bir veri seti seçin (CSV)", type="csv")
            if uploaded_file_csv is not None:
                veri_analizi(uploaded_file_csv)
            else:
                st.write('Lütfen veri yükleyin...')
                

        with tab2:
            uploaded_file_xlsx = st.file_uploader("Lütfen bir veri seti seçin (XLSX)", type="xlsx")
            if uploaded_file_csv is not None:
                veri_analizi(uploaded_file_csv)
            else:
                st.write('Lütfen veri yükleyin...')

        with tab3:
            uploaded_file = st.file_uploader("Lütfen bir veri seti seçin (TXT)", type="txt")
            if uploaded_file_csv is not None:
                veri_analizi(uploaded_file_csv)
            else:
                st.write('Lütfen veri yükleyin...')"""
        ###########################################################################################################
        with tab1:
            uploaded_file_csv = st.file_uploader("Lütfen bir veri seti seçin (CSV)", type="csv")
            if uploaded_file_csv is not None:
                veri_analizi(uploaded_file_csv)
            else:
                st.write('Lütfen veri yükleyin...')

        with tab2:
            uploaded_file_xlsx = st.file_uploader("Lütfen bir veri seti seçin (XLSX)", type="xlsx")
            if uploaded_file_xlsx is not None:
                veri_analizi(uploaded_file_xlsx)
            else:
                st.write('Lütfen veri yükleyin...')

        with tab3:
            uploaded_file_txt = st.file_uploader("Lütfen bir veri seti seçin (TXT)", type="txt")
            if uploaded_file_txt is not None:
                veri_analizi(uploaded_file_txt)
            else:
                st.write('Lütfen veri yükleyin...')
 
    