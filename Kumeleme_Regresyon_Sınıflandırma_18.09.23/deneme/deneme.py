import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from dateutil import parser
from islemler.regresyonAnaliziFarkliTrainVeTestSet import regresyon_analizi_sonuclar_train


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor


sns.set()


def regresyon_analizi(uploaded_file):
 # Başlık
 st.write("Regresyon Analizi işlemleri")

 df = pd.read_csv(uploaded_file)
 st.write("Seçilen veri seti:")
 st.dataframe(df)

 
 for col in df.columns:
            try:
                # Tarih sütunu olup olmadığını belirlemek için bir örnek veri kullanın
                sample = df[col].dropna().iloc[0]
                parser.parse(sample)

                # Tarih sütunu ise, tarih veri tipine dönüştürün
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                df['gün'] = df[col].dt.day
                df['ay'] = df[col].dt.month
                df['yıl'] = df[col].dt.year
                df = df.drop(columns=col)
                

            except (TypeError, ValueError):
                # Tarih sütunu değilse, bir sonraki sütuna geçin
                continue

 
 # Hedef değişken seçimi
 column_name = st.text_input("Lütfen  hedef değişkeni seçiniz")
 if column_name:
  if column_name in df.select_dtypes(include=[np.number]).columns:
   # Veri seti önizleme
   if st.checkbox('Veri setini tablo formatında göster'):
     st.dataframe(df)


   # Bütün değişken isimlerini göster
   if st.checkbox('Değişken isimlerini göster'):
     st.markdown(
         list(df.columns)
           )


   # Plot the relation between target and explanatory variables
   # when the checkbox is ON.
   if st.checkbox('Plot the relation between target and explanatory variables'):
     # Select one explanatory variable for ploting
     checked_variable = st.selectbox(
       'Select one explanatory variable:',
       df.drop(columns= column_name).columns
       )
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

   # Kullanmak istemediğiniz değişkenleri seçiniz
  Features_chosen = []
  Features_NonUsed = st.multiselect(
    'Kullanmak istemediğiniz değişkenleri seçiniz', 
    df.drop(columns= column_name).columns
    )

  # Seçilen sütunlar çıkarılır
  df = df.drop(columns=Features_NonUsed)


  # Choose whether you will perform logarithmic transformation
  left_column, right_column = st.columns(2)
  bool_log = left_column.radio(
        'Logaritmik transformasyon uygulamak istiyor musunuz?', 
        ('Hayır','Evet')
        )

  df_log, Log_Features = df.copy(), []
  if bool_log == 'Yes':
    Log_Features = right_column.multiselect(
            'Logaritmik transformasyon uygulamak istediğiniz değişkenleri seçiniz.',
            df.columns
            )
    # Logaritmik transformasyon uygulaması
    df_log[Log_Features] = np.log(df_log[Log_Features])


  # Standardizasyon işlemleri
  left_column, right_column = st.columns(2)
  bool_std = left_column.radio(
        'Standardizasyon kullanmak istiyor musunuz?',
        ('Hayır','Evet')
        )

  df_std = df_log.copy()
  if bool_std == 'Yes':
    Std_Features_NotUsed = right_column.multiselect(
            'Standardizasyon kullanmak istemediğiniz değişkenleri seçiniz.', 
            df_log.drop(columns=[column_name]).columns
            )
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


  # Veri kümesini bölme
  X_train, X_val, Y_train, Y_val = train_test_split(
    df_std.drop(columns=[column_name]), 
    df_std[column_name], 
    test_size=test_size, 
    random_state=random_seed
    )


  # Bir doğrusal regresyon örneği oluşturun
  regressor = LinearRegression()
  regressor.fit(X_train, Y_train)
 
  Y_pred_train = regressor.predict(X_train)
  Y_pred_val = regressor.predict(X_val)

  # Perform inverse conversion if the logarithmic transformation was performed.
  if column_name in Log_Features:
    Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
    Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)



  """
  ## Sonuçları göster
  """

  """
  ### Modelin doğruluğu
  """
  R2 = r2_score(Y_val, Y_pred_val)
  st.write(f'R2 değeri: {R2:.2f}')


  """
  ### Sonuçları çiz
  """
  left_column, right_column = st.columns(2)
  show_train = left_column.radio(
          'Eğitim veri kümesinin sonucunu oluştur:', 
          ('Evet','Hayır')
          )
  show_val = right_column.radio(
          'Doğrulama veri kümesinin sonucunu çizin:', 
          ('Evet','Hayır')
          )


  # Get the maximum value of all objective variable data,
  # including predicted values
  y_max_train = max([max(Y_train), max(Y_pred_train)])
  y_max_val = max([max(Y_val), max(Y_pred_val)])
  y_max = int(max([y_max_train, y_max_val])) 


  # Allows the axis range to be changed dynamically
  left_column, right_column = st.columns(2)
  x_min = left_column.number_input('x_min:',value=0,step=1)
  x_max = right_column.number_input('x_max:',value=y_max,step=1)
  left_column, right_column = st.columns(2)
  y_min = left_column.number_input('y_min:',value=0,step=1)
  y_max = right_column.number_input('y_max:',value=y_max,step=1)


  # Show the results
  fig = plt.figure(figsize=(3, 3))
  if show_train == 'Evet':
    plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
  if show_val == 'Evet':
    plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validation data")

  plt.xlabel(column_name,fontsize=8)
  plt.ylabel("Prediction of "+column_name, fontsize=8)
  plt.xlim(int(x_min), int(x_max)+5)
  plt.ylim(int(y_min), int(y_max)+5)
  plt.legend(fontsize=6)
  plt.tick_params(labelsize=6)

  # Display by Streamlit
  st.pyplot(fig)

def regresyon_analizi_sonuclar():
        tab1, tab2, tab3 = st.tabs(["CSV", "XLSX", "TXT"])
        with tab1:
            if st.checkbox('Train ve Test setlerini birlikte eklemek istiyor musunuz?'):
             uploaded_file_csv = st.file_uploader("Lütfen bir veri seti seçin (CSV)", type="csv")
             if uploaded_file_csv is not None:
                regresyon_analizi(uploaded_file_csv)
             else:
                st.write('Lütfen veri yükleyin...')

            if st.checkbox('Train ve Test setlerini ayrı eklemek istiyor musunuz?'):
             regresyon_analizi_sonuclar_train()

            

        with tab2:
            uploaded_file_xlsx = st.file_uploader("Lütfen bir veri seti seçin (XLSX)", type="xlsx")
            if uploaded_file_xlsx is not None:
                regresyon_analizi(uploaded_file_xlsx)
            else:
                st.write('Lütfen veri yükleyin...')

        with tab3:
            uploaded_file_txt = st.file_uploader("Lütfen bir veri seti seçin (TXT)", type="txt")
            if uploaded_file_txt is not None:
                regresyon_analizi(uploaded_file_txt)
            else:
                st.write('Lütfen veri yükleyin...')


                
                """

  left_column, right_column = st.columns(2)
  show_train = left_column.radio(
          'Eğitim veri kümesinin sonucunu oluştur:', 
          ('Evet','Hayır')
          )
  show_val = right_column.radio(
          'Doğrulama veri kümesinin sonucunu çizin:', 
          ('Evet','Hayır')
          )


  # Get the maximum value of all objective variable data,
  # including predicted values
  y_max_train = max([max(Y_train), max(Y_pred_train)])
  y_max_val = max([max(Y_val), max(Y_pred_val)])
  y_max = int(max([y_max_train, y_max_val])) 


  # Allows the axis range to be changed dynamically
  left_column, right_column = st.columns(2)
  x_min = left_column.number_input('x_min:',value=0,step=1)
  x_max = right_column.number_input('x_max:',value=y_max,step=1)
  left_column, right_column = st.columns(2)
  y_min = left_column.number_input('y_min:',value=0,step=1)
  y_max = right_column.number_input('y_max:',value=y_max,step=1)


  # Show the results
  fig = plt.figure(figsize=(3, 3))
  if show_train == 'Evet':
    plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
  if show_val == 'Evet':
    plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validation data")

  plt.xlabel(column_name,fontsize=8)
  plt.ylabel("Prediction of "+column_name, fontsize=8)
  plt.xlim(int(x_min), int(x_max)+5)
  plt.ylim(int(y_min), int(y_max)+5)
  plt.legend(fontsize=6)
  plt.tick_params(labelsize=6)

  # Display by Streamlit
  st.pyplot(fig)


    # farklı değişken sayısına göre encoding işlemi yap
  for i, sütun in enumerate(sütun_isimleri):
    if farklı_değişkenler[i] < 10:
        # Dummy encoding
        df = pd.get_dummies(df, columns=[sütun])
    else:
        # One-Hot encoding
        df = pd.get_dummies(df, columns=[sütun], prefix='', prefix_sep='')
        
        
  """
                
