import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from dateutil import parser
import streamlit.components.v1 as components  
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from islemler.veriAnalizi import veri_analizi, veri_analizi_sonuclar
from islemler.regresyonAnalizi import regresyon_analizi, regresyon_analizi_sonuclar
from islemler.siniflandirmaAnalizi import siniflandirma_analizi, siniflandirma_analizi_sonuclar
from islemler.kumelemeAnalizi import kumeleme_analizi, kumeleme_analizi_sonuclar

image = Image.open('logo.png')
st.image(image,width=200)
st.title(':blue[**TEAM-MUTFAK Analitik Alanı**]')

# Butonları oluşturmak için aşağıdaki kodu kullanabilirsiniz
option = st.sidebar.selectbox("Lütfen bir seçim yapınız:", ("Anasayfa","Veri Analizi", "Regresyon Analizi", "Sınıflandırma Analizi", "Kümeleme Analizi"))

# Seçilen butona göre mesaj göstermek için aşağıdaki kodu kullanabilirsiniz
if option == "Anasayfa":
 # Giriş sayfasını oluşturmak için aşağıdaki kodu kullanabilirsiniz
 st.markdown(':blue[TEAM-MUTFAK Analitik Alanına Hoş Geldiniz!]')
 st.markdown("Bu uygulama **TEAM-MUTFAK** şirketi tarafından oluşturulmuştur ve çeşitli analitik işlemler yapmak için tasarlanmıştır.")

 optionMain = st.selectbox(
    'Aşağıdan Seçim Yapabilirsiniz',
    ('Seçenekler','Veri Analizi', 'Regresyon Analizi', 'Sınıflandırma Analizi', 'Kümeleme Analizi'))


 if(optionMain == 'Veri Analizi'):
     #st.write('You selected:', optionMain)
     veri_analizi_sonuclar()
 elif(optionMain == 'Regresyon Analizi'):
     #st.write('You selected:', optionMain)
     st.markdown("**Regresyon analizi**, bağımlı değişken ile bağımsız değişken arasındaki ilişkiyi tahmin etmek için kullanılan istatistiksel bir araçtır. Daha spesifik olarak, bağımlı değişkenin bağımsız değişkenlerdeki değişikliklere göre nasıl değiştiğine odaklanır. Ayrıca değişkenler arasındaki gelecekteki ilişkinin modellenmesine de yardımcı olur. İki değişken varsa, tahminin temelini oluşturan değişken bağımsız değişkendir. Değeri tahmin edilecek değişken bağımlı değişken olarak bilinir. ")
     regresyon_analizi_sonuclar()
 elif(optionMain == 'Sınıflandırma Analizi'):
     #st.write('You selected:', optionMain)
     siniflandirma_analizi_sonuclar()
 elif(optionMain == 'Kümeleme Analizi'):
     #st.write('You selected:', optionMain)
     kumeleme_analizi_sonuclar()


elif option == "Veri Analizi":
     st.header(':blue[**Veri Analizi**]')
     veri_analizi_sonuclar()

elif option == "Regresyon Analizi":
     st.header(':blue[**Regresyon Analizi**]')
     st.markdown("**Regresyon analizi**, bağımlı değişken ile bağımsız değişken arasındaki ilişkiyi tahmin etmek için kullanılan istatistiksel bir araçtır. Daha spesifik olarak, bağımlı değişkenin bağımsız değişkenlerdeki değişikliklere göre nasıl değiştiğine odaklanır. Ayrıca değişkenler arasındaki gelecekteki ilişkinin modellenmesine de yardımcı olur. İki değişken varsa, tahminin temelini oluşturan değişken bağımsız değişkendir. Değeri tahmin edilecek değişken bağımlı değişken olarak bilinir. ")
     regresyon_analizi_sonuclar()
     
elif option == "Sınıflandırma Analizi":
     st.header(':blue[**Sınıflandırma Analizi**]')
     siniflandirma_analizi_sonuclar()

else:
     st.header(':blue[**Kümeleme Analizi**]')
     kumeleme_analizi_sonuclar()









