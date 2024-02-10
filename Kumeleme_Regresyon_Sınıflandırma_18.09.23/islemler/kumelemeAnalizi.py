import streamlit as st
import pandas as pd 
import numpy as np 
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from dateutil import parser
import io


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
import seaborn as sns
import random




def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgb({r}, {g}, {b})"

def elbow_method(data, max_clusters=10):
    distortions = []
    K = range(1, max_clusters)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(data)
        distortions.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
    return distortions

def silhouette_analysis(data, max_clusters=10):
    silhouette_scores = []
    K = range(2, max_clusters)
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(data)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(data, labels))
    return silhouette_scores


def aic_bic(data, max_clusters=10):
    aic_scores = []
    bic_scores = []
    K = range(1, max_clusters)
    for k in K:
        gmm = GaussianMixture(n_components=k).fit(data)
        aic_scores.append(gmm.aic(data))
        bic_scores.append(gmm.bic(data))
    return aic_scores, bic_scores            

def gap_statistic(data, max_clusters=10, n_refs=5):
    gaps = np.zeros((len(range(1, max_clusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})

    for gap_index, k in enumerate(range(1, max_clusters)):
        ref_disps = np.zeros(n_refs)
        for i in range(n_refs):
            random_data = np.random.random_sample(size=data.shape)
            km = KMeans(n_clusters=k)
            km.fit(random_data)
            ref_disps[i] = np.log(km.inertia_)

        km = KMeans(n_clusters=k)
        km.fit(data)
        orig_disp = np.log(km.inertia_)
        gap = np.mean(ref_disps) - orig_disp
        gaps[gap_index] = gap

    return gaps



def kumeleme_analizi(uploaded_file):
    st.write("Kümeleme Analizi işlemleri")

    if uploaded_file is not None:
        content = uploaded_file.getvalue()
        try:
            data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        except:
            try:
                data = pd.read_excel(uploaded_file)
            except:
                data = pd.read_table(io.StringIO(content.decode('utf-8')))
        st.write(data.head())

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            st.error("Veri setinde numerik türde değişken bulunmamaktadır.")
        elif len(non_numeric_cols) > 0:
            st.warning("Kümeleme analizinde değişkenler ya komple numerik ya da komple kategorik olmalıdır.")
            st.warning("Numerik olmayan değişkenler: {}".format(non_numeric_cols))
        else:
            
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            distortions = elbow_method(data_scaled)
            silhouette_scores = silhouette_analysis(data_scaled)
            gap_scores = gap_statistic(data_scaled)
            aic_scores, bic_scores = aic_bic(data_scaled)

            st.markdown("<span style='font-size:18px;font-weight:bold;color:black'>Optimum Küme Sayısını Belirlemek</span> <span style='font-size:18px;font-weight:bold'>kümeleme çalışmalarında çok kritikdir. Bunun için belirli yöntemler bulunmaktadır. <span style='color:red'>Elbow</span>, <span style='color:blue'>Silhouette</span>, <span style='color:brown'>Gap Statistic</span>, <span style='color:purple'>AIC</span> ve <span style='color:purple'>BIC</span> yöntemleri bunlara örnektir. Bunların veri üzerinde ölçülmesini ve önermesini görelim.</span>", unsafe_allow_html=True)

            
            st.write("------------------------------------------------------------------------------------------------------")
            st.write("------------------------------------------------------------------------------------------------------")
            
            st.markdown("<span style='font-size:18px;font-weight:bold;color:red'>Elbow</span> <span style='font-size:18px;font-weight:bold'>yönteminde Distortion'un düşüş hızının azaldığı nokta en uygun küme   sayısı olarak kabul edilir.</span>", unsafe_allow_html=True)


            plt.figure(figsize=(10, 24))
            plt.subplot(4, 1, 1)
            plt.plot(range(1, len(distortions) + 1), distortions, 'bx-')
            plt.xlabel('Küme Sayısı')
            plt.ylabel('Distortion')
            plt.title('Elbow Method')
            plt.tight_layout()
            st.pyplot(plt)
            
            st.write("------------------------------------------------------------------------------------------------------")
            st.markdown("<span style='font-size:18px;font-weight:bold;color:blue'>Silhouette</span> <span style='font-size:18px;font-weight:bold'>analizinde en yüksek Silhouette skoruna sahip olan küme sayısı en uygun küme sayısı olarak kabul edilir.</span>", unsafe_allow_html=True)
            plt.figure(figsize=(10, 24))
            plt.subplot(4, 1, 2)
            plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'bx-')
            plt.xlabel('Küme Sayısı')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Analysis')
            plt.tight_layout()
            st.pyplot(plt)
            
            st.write("------------------------------------------------------------------------------------------------------")
            st.markdown("<span style='font-size:18px;font-weight:bold;color:brown'>Gap Statistic</span> <span style='font-size:18px;font-weight:bold'>yönteminde en yüksek Gap skoruna sahip olan küme sayısı en uygun küme sayısı olarak kabul edilir.</span>", unsafe_allow_html=True)
            plt.figure(figsize=(10, 24))
            plt.subplot(4, 1, 3)
            plt.plot(range(1, len(gap_scores) + 1), gap_scores, 'bx-')
            plt.xlabel('Küme Sayısı')
            plt.ylabel('Gap Score')
            plt.title('Gap Statistic')
            plt.tight_layout()
            st.pyplot(plt)
            
            
            st.write("------------------------------------------------------------------------------------------------------")
            st.markdown("<span style='font-size:18px;font-weight:bold;color:purple'>AIC ve BIC</span> <span style='font-size:18px;font-weight:bold'>yöntemlerinde skorların düşüş hızının azaldığı ve en düşük değerlere ulaştığı nokta en uygun küme sayısı olarak kabul edilir.</span>", unsafe_allow_html=True)
            plt.figure(figsize=(10, 24))
            plt.subplot(4, 1, 4)
            plt.plot(range(1, len(aic_scores) + 1), aic_scores, 'bx-', label='AIC')
            plt.plot(range(1, len(bic_scores) + 1), bic_scores, 'rx-', label='BIC')
            plt.xlabel('Küme Sayısı')
            plt.ylabel('Scores')
            plt.title('AIC/BIC Scores')
            plt.legend()

            plt.tight_layout()
            st.pyplot(plt)
            
            kumeleme_yontemleri = {
                "K-means": "Küme merkezlerini optimize ederek belirli bir sayıda küme oluşturur.",
                "DBSCAN": "Yoğun bölgeleri belirleyerek kümeler oluşturur ve gürültüyü ayırt eder.",
                "Agglomerative Hierarchical Clustering": "Birleştirici yöntemle aşamalı olarak kümeleri oluşturur.",
                "Divisive Hierarchical Clustering": "Bölücü yöntemle aşamalı olarak kümeleri oluşturur.",
                "Gaussian Mixture Models (GMM)": "Veri noktalarının olasılık dağılımlarına göre kümeler oluşturur.",
                "Spectral Clustering": "Graf teorisi kullanarak veri noktalarını kümeler halinde gruplar.",
                "Affinity Propagation": "Veri noktaları arasındaki benzerliği kullanarak tüm veri noktalarını temsilcileriyle eşleştirir.",
                "OPTICS": "Yoğunluk tabanlı bir yöntemle, değişken yoğunluklu bölgelerde de kümeler oluşturur.",
                "BIRCH": "Küme merkezleri ve dağılımları hakkında özet bilgiler sağlayarak veriyi sıkıştırır ve işler.",
                "Mean Shift": "Veri noktalarının yerel maksimum yoğunluğa doğru kaydırılmasıyla kümeler oluşturur."
            }

            st.markdown("<span style='font-size:18px;font-weight:bold;color:black'>Aşağıda kümeleme yöntemleri ve kısa açıklamaları bulunmaktadır. Hangi yöntemi kullanmak istediğinizi seçebilirsiniz.</span>", unsafe_allow_html=True)

            for yontem, aciklama in kumeleme_yontemleri.items():
                renk = random_color()
                st.markdown(f"<span style='font-size:18px;font-weight:bold;color:{renk}'>{yontem}</span> <span style='font-size:18px;font-weight:bold'>: {aciklama}</span>", unsafe_allow_html=True)
                
            optimum_kume_sayisi = st.number_input("Lütfen görselleri inceleyerek belirlemek istediğiniz küme sayısını girin:", min_value=1, value=3, step=1)    
            
            
            secilen_yontem = st.selectbox("Hangi kümeleme yöntemini kullanmak istiyorsunuz?", options=list(kumeleme_yontemleri.keys()))
            
            st.write(f"Seçilen yöntem: {secilen_yontem}")
            
            
               
            if secilen_yontem == "K-means":
                model = KMeans(n_clusters=optimum_kume_sayisi)
            elif secilen_yontem == "DBSCAN":
                model = DBSCAN(eps=0.5)
            elif secilen_yontem == "Agglomerative Hierarchical Clustering":
                model = AgglomerativeClustering(n_clusters=optimum_kume_sayisi)
            elif secilen_yontem == "Divisive Hierarchical Clustering":
                model = AgglomerativeClustering(n_clusters=optimum_kume_sayisi, linkage='ward', compute_full_tree=True)
            elif secilen_yontem == "Gaussian Mixture Models (GMM)":
                model = GaussianMixture(n_components=optimum_kume_sayisi)
            elif secilen_yontem == "Spectral Clustering":
                model = SpectralClustering(n_clusters=optimum_kume_sayisi, assign_labels="discretize", random_state=0)
            elif secilen_yontem == "Affinity Propagation":
                model = AffinityPropagation(random_state=5)
            elif secilen_yontem == "OPTICS":
                model = OPTICS(min_samples=2)
            elif secilen_yontem == "BIRCH":
                model = Birch(n_clusters=optimum_kume_sayisi)
            elif secilen_yontem == "Mean Shift":
                model = MeanShift()

            if secilen_yontem != "Gaussian Mixture Models (GMM)":
                y_kume = model.fit_predict(data_scaled)
            else:
                y_kume = model.fit(data_scaled).predict(data_scaled)

            data["Küme"] = y_kume

            # Küme sonuçlarını anlamlandırıcı şekilde göster
            st.write("Kümeleme sonuçları:")

            if secilen_yontem != "DBSCAN" and secilen_yontem != "OPTICS":
                for i in range(optimum_kume_sayisi):
                    st.write(f"Küme {i}: {len(data[data['Küme'] == i])} veri noktası")
            else:
                for i in set(y_kume):
                    if i == -1:
                        st.write(f"Gürültü: {len(data[data['Küme'] == i])} veri noktası")
                    else:
                        st.write(f"Küme {i}: {len(data[data['Küme'] == i])} veri noktası")

            st.write(data.head(10))

            # Grafiksel olarak göster
            if data_scaled.shape[1] > 2:
                pca = PCA(n_components=2)
                data_pca = pca.fit_transform(data_scaled)
                data["PCA1"] = data_pca[:, 0]
                data["PCA2"] = data_pca[:, 1]
            else:
                data["PCA1"] = data_scaled[:, 0]
                data["PCA2"] = data_scaled[:, 1]

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x="PCA1", y="PCA2", data=data, hue="Küme", palette="viridis", legend="full")
            plt.xlabel("İlk Bileşen")
            plt.ylabel("İkinci Bileşen")
            plt.title(f"{secilen_yontem} ile Kümeleme Analizi Sonucu")
            st.pyplot(plt)
            
            
            
            

def kumeleme_analizi_sonuclar():
        st.write("""
                Kümeleme analizi, veri noktalarını benzer özelliklere sahip gruplara ayırmak için kullanılan bir veri madenciliği tekniğidir. 
                Bu uygulamada, çeşitli kümeleme yöntemlerini kullanarak veri setinizi analiz edebilirsiniz.
              """)
        tab1, tab2, tab3 = st.tabs(["CSV", "XLSX", "TXT"])

        with tab1:
            uploaded_file_csv = st.file_uploader("Lütfen bir veri seti seçin (CSV)", type=["csv", "text/csv"])
            if uploaded_file_csv is not None:
                kumeleme_analizi(uploaded_file_csv)
            else:
                st.write('Lütfen veri yükleyin...')

        with tab2:
            uploaded_file_xlsx = st.file_uploader("Lütfen bir veri seti seçin (XLSX)", type=["xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"])
            if uploaded_file_xlsx is not None:
                kumeleme_analizi(uploaded_file_xlsx)
            else:
                st.write('Lütfen veri yükleyin...')

        with tab3:
            uploaded_file_txt = st.file_uploader("Lütfen bir veri seti seçin (TXT)", type=["txt", "text/plain"])
            if uploaded_file_txt is not None:
                kumeleme_analizi(uploaded_file_txt)
            else:
                st.write('Lütfen veri yükleyin...')
 