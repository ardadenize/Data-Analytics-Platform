import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from dateutil import parser
from sklearn.metrics import mean_squared_error
from islemler.regresyonModelleri import regresyon_model



sns.set()


def regresyon_analizi(uploaded_file_csv_train, uploaded_file_csv_test):

 # Başlık
 st.write("Regresyon Analizi işlemleri")
 dftrain = pd.read_csv(uploaded_file_csv_train)  
 dftest = pd.read_csv(uploaded_file_csv_test)


 # Train ve Test setlerini birleştir
 df = pd.concat([dftrain, dftest], keys=['TrainSet', 'TestSet'])

 st.write("Seçilen **Train** veri seti:")
 st.dataframe(dftrain)
 st.write("Seçilen **Test** veri seti:")
 st.dataframe(dftest)


 for col in df.columns:
    try:
      # Tarih sütunu olup olmadığını belirlemek için bir örnek veri kullanın
      sample = df[col].dropna().iloc[0]
      parser.parse(sample)

      # Tarih sütunu ise, tarih veri tipine dönüştürün
      df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
      df['Gün'] = df[col].dt.day
      df['Ay'] = df[col].dt.month
      df['Yıl'] = df[col].dt.year
      df = df.drop(columns=col)
                

    except (TypeError, ValueError):
      # Tarih sütunu değilse, bir sonraki sütuna geçin
      continue


 #numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
 #categoric_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

 # Numeric olan değişkenlerin seçimi
 numeric_variables = st.multiselect('Numerik olan sütunları seçiniz', df.columns)
 categoric_variables = df.drop(columns= numeric_variables).columns
 

 # Numeric değişkenlerin içerisinde kategorik değişken varsa değiştir
 is_cat = pd.api.types.is_categorical_dtype(df[numeric_variables])
 if is_cat:
    df[numeric_variables] = pd.factorize(df[numeric_variables])[0]
    return numeric_variables
  
 st.write("Numerik Değişkenler:", numeric_variables)
 st.write("Kategorik Değişkenler:", categoric_variables)
 
 
 # Hedef değişken seçimi
 Features_chosen1 = []
 column_name = st.selectbox('Lütfen  hedef değişkeni seçiniz', df.drop(columns= categoric_variables).columns)
 if column_name:
  if column_name in df.select_dtypes(include=[np.number]).columns:
   # Veri seti önizleme
   # Değişkenlerin tipini yazdır
   if st.checkbox('Veri setini tablo formatında göster'):
     st.dataframe(df)


   # Bütün değişken isimlerini göster
   if st.checkbox('Değişken isimlerini göster'):
     st.markdown(list(df.columns))


   # Hedef ve açıklayıcı değişkenler arasındaki ilişkiyi çizin
   if st.checkbox('Hedef ve açıklayıcı değişkenler arasındaki ilişkiyi çizin'):
     # Select one explanatory variable for ploting
     checked_variable = st.selectbox('Select one explanatory variable:', df.drop(columns= column_name).columns)

     # Plot
     fig, ax = plt.subplots(figsize=(5, 3))
     ax.scatter(x=df[checked_variable], y=df[column_name])
     plt.xlabel(checked_variable)
     plt.ylabel(column_name)
     st.pyplot(fig)
  else:
     st.write(f"Hata: {column_name}")


  """
  ## Preprocessing
  """

  # Encoding işlemleri
  sütun_isimleri = df.drop(columns= numeric_variables).columns

  # her bir sütunun içindeki farklı değişken sayısını bul
  farklı_değişkenler = [len(df[sütun].unique()) for sütun in sütun_isimleri]

  # farklı değişken sayısına göre encoding işlemi yap
  for i, sütun in enumerate(sütun_isimleri):
    if farklı_değişkenler[i] < 100:
        # Dummy encoding
        df = pd.get_dummies(df, columns=[sütun])
    else:
        # One-Hot encoding
        df = pd.get_dummies(df, columns=[sütun], prefix='', prefix_sep='')
        
  # Kullanmak istemediğiniz değişkenleri seçiniz
  Features_chosen = []
  Features_NonUsed = st.multiselect('Kullanmak istemediğiniz değişkenleri seçiniz', df.drop(columns= column_name).columns)
  # Seçilen sütunlar çıkarılır
  df = df.drop(columns=Features_NonUsed)



  st.write(df)


  # Logaritmik transformasyon veya Standardizasyon
  left_column, right_column = st.columns(2)
  bool_log = left_column.radio(
        'Logaritmik transformasyon uygulamak istiyor musunuz?', 
        ('Logaritmik transformasyon veya Standardizasyon kullanmak istemiyorum','Logaritmik transformasyon kullanmak istiyorum','Standardizasyon kullanmak istiyorum'))

  df_log, Log_Features = df.copy(), []
  if bool_log == 'Logaritmik transformasyon kullanmak istiyorum':
    Log_Features = right_column.multiselect(
            'Logaritmik transformasyon uygulamak istediğiniz değişkenleri seçiniz.', df.columns)
    

    # Logaritmik transformasyon uygulaması
    df_log[Log_Features] = np.log(df_log[Log_Features])
  elif bool_log == 'Standardizasyon kullanmak istiyorum':
    df_std = df_log.copy()
    Std_Features_NotUsed = right_column.multiselect(
            'Standardizasyon kullanmak istemediğiniz değişkenleri seçiniz.', df_log.drop(columns=[column_name]).columns)
    

    # explanatory variables ataması , 
    # excluded of ones in "Std_Features_NotUsed",
    # to "Std_Features_chosen"
    Std_Features_chosen = []
    for name in df_log.drop(columns=[column_name]).columns:
      if name in Std_Features_NotUsed:
        continue
      else:
        Std_Features_chosen.append(name)


    # Standardizasyon uygulanması
    sscaler = preprocessing.StandardScaler()
    sscaler.fit(df_std[Std_Features_chosen])
    df_std[Std_Features_chosen] = sscaler.transform(df_std[Std_Features_chosen])


  """
  ### Veri setini eğitim ve doğrulama veri setlerine ayırın
  """

  left_column, right_column = st.columns(2)
  test_size = left_column.number_input(
          'Test Seti veri boyutu(min: 0.0, max: 1.0):',
          min_value=0.0,
          max_value=1.0,
          value=0.2,
          step=0.1,
           )
  
  random_seed = right_column.number_input(
                'Random seed(Nonnegative integer):',
                  value=0, 
                  step=1,
                  min_value=0)

  df_train = df.loc['TrainSet']
  df_test = df.loc['TestSet']

  # Veri kümesini bölme

# Split into 
  x = df_train.drop(columns=[column_name])
  y = df_train[column_name]
  X_train,X_val,Y_train,Y_val = train_test_split(x,y,test_size=test_size, random_state=random_seed)

  test_set_drop = df_test.drop(columns=[column_name])
  test_target = df_test[column_name]

  
  regresyon_model(X_train, X_val, Y_train, Y_val, test_set_drop, test_target)



def regresyon_analizi_sonuclar_train():
        tab1, tab2, tab3 = st.tabs(["CSV", "XLSX", "TXT"])
        with tab1:
            uploaded_file_csv_train = st.file_uploader("Lütfen bir **Train** veri seti seçin (CSV)", type="csv")
            
            if uploaded_file_csv_train is not None:
                uploaded_file_csv_test = st.file_uploader("Lütfen bir **Test** veri seti seçin (CSV)", type="csv")
                if uploaded_file_csv_test is not None:
                 regresyon_analizi(uploaded_file_csv_train, uploaded_file_csv_test)
                else:
                 st.write('Lütfen veri yükleyin...')
            else:
                st.write('Lütfen veri yükleyin...')

        with tab2:
            uploaded_file_xlsx_train = st.file_uploader("Lütfen bir **Train** veri seti seçin (XLSX)", type="xlsx")
            uploaded_file_xlsx_test = st.file_uploader("Lütfen bir **Test** veri seti seçin (XLSX)", type="xlsx")
            if uploaded_file_xlsx_train is not None and uploaded_file_xlsx_test is not None:
                regresyon_analizi(uploaded_file_xlsx_train, uploaded_file_xlsx_test)
            else:
                st.write('Lütfen veri yükleyin...')

        with tab3:
            uploaded_file_txt_train = st.file_uploader("Lütfen bir **Train** veri seti seçin (TXT)", type="txt")
            uploaded_file_txt_test = st.file_uploader("Lütfen bir **Test** veri seti seçin (TXT)", type="txt")
            if uploaded_file_txt_train is not None and uploaded_file_txt_test is not None:
                regresyon_analizi(uploaded_file_txt_train, uploaded_file_txt_test)
            else:
                st.write('Lütfen veri yükleyin...')
