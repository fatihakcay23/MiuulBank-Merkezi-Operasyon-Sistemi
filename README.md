# MiuulBank Merkezi Operasyon Sistemi

Kredi kartı müşteri kaybı (churn) verisi üzerine kurulu, çok modüllü bir CRM ve karar destek sistemi. SQL tabanlı raporlama, makine öğrenmesi ile churn tahmini, davranışsal benzerlik analizine dayalı risk skorlama ve istatistiksel hipotez testlerini tek bir Streamlit uygulamasında birleştirir.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Veri Seti](#veri-seti)
- [Mimari](#mimari)
- [Modüller](#modüller)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Bilinen Sınırlamalar](#bilinen-sınırlamalar)
- [Geliştirme Fikirleri](#geliştirme-fikirleri)
- [Lisans](#lisans)

## Genel Bakış

Proje, kurgusal bir banka ("MiuulBank") için müşteri kaybı riskini azaltmayı hedefleyen üç ayrı analiz/aksiyon modülünü tek bir arayüzde bir araya getirir:

1. Yönetici düzeyinde KPI takibi ve SQL tabanlı stratejik raporlama, RandomForest ile churn tahmini
2. Müşteri bazlı, KNN benzerlik analizine dayalı proaktif risk skorlama ve aksiyon önerisi
3. A/B testleri ve K-Means segmentasyonu ile istatistiksel keşifsel analiz

## Veri Seti

Kaynak veri seti [Kaggle Credit Card Customers (BankChurners.csv)](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) veri setidir. 10.127 müşteri kaydı ve 21 temel değişken içerir; ayrıca veri setiyle birlikte gelen, modelleme için kullanılmaması gereken 2 adet önceden hesaplanmış Naive Bayes sütunu bulunur (bu sütunlar tüm modüllerde ayıklanır).

Öne çıkan değişkenler:

| Sütun | Açıklama |
|---|---|
| `Attrition_Flag` | Hedef değişken — `Existing Customer` / `Attrited Customer` |
| `Customer_Age`, `Gender`, `Dependent_count` | Demografik bilgiler |
| `Income_Category`, `Card_Category` | Gelir aralığı ve kart tipi |
| `Months_on_book`, `Total_Relationship_Count` | Bankayla ilişki süresi ve ürün sayısı |
| `Months_Inactive_12_mon`, `Contacts_Count_12_mon` | Aktiflik ve iletişim sıklığı |
| `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Utilization_Ratio` | Kredi limiti ve kullanım oranı |
| `Total_Trans_Amt`, `Total_Trans_Ct` | Toplam harcama tutarı ve işlem adedi |

Veri setinde churn oranı dengesizdir (yaklaşık %16 Attrited Customer); bu nedenle model değerlendirmesinde yalnızca doğruluk (accuracy) değil precision/recall metrikleri de raporlanır.

## Mimari

Uygulama tek bir Streamlit giriş noktası (`app.py`) üzerinden, sol menüden seçilen üç bağımsız modülü render eden fonksiyonel bir yapı kullanır:

```
main()
 ├── app_sql_analytics()      # Modül 1
 ├── app_proactive_banker()   # Modül 2
 └── app_customer_stats()     # Modül 3
```

Modül 1, veriyi bir MS SQL Server veritabanından okurken; Modül 2 ve Modül 3, doğrudan yerel bir CSV dosyasından (`BankChurners.csv` veya `BankChurners_Cleaned.csv`) çalışacak şekilde tasarlanmıştır. Bu nedenle Modül 2 ve 3, SQL Server kurulumu olmadan da doğrudan çalıştırılabilir.

## Modüller

### Modül 1 — SQL & AI Analytics

- `pyodbc` ile yerel bir MS SQL Server örneğine bağlanır ve `BankChurners_Cleaned` tablosunu sorgular.
- Yönetici özeti (KPI) ekranı: toplam müşteri, churn oranı, ortalama kredi limiti ve harcama.
- Önceden tanımlanmış SQL senaryolarından oluşan bir raporlama merkezi: erken uyarı/risk sorguları (ör. yüksek limitli ama uzun süredir inaktif müşteriler), satış/büyüme fırsatları (ör. limit artış adayları, kart yükseltme adayları) ve operasyonel analiz sorguları.
- `RandomForestClassifier` ile churn tahmin modeli: kullanıcı bir müşteri profili girer, model terk etme olasılığını hesaplar; sonuç bir gauge grafiği, özellik önem sıralaması ve gerçek test seti metrikleriyle (accuracy, precision, recall) birlikte sunulur.

### Modül 2 — Proaktif Bankacı (Risk) Ekranı

- Kendi içinde uçtan uca bir özellik mühendisliği hattı çalıştırır: kategorik/nümerik sütun ayrımı, IQR tabanlı aykırı değer baskılama (`outlier_thresholds` / `replace_with_thresholds`), ikili ve çok sınıflı kategorik değişkenler için label/one-hot encoding, `MinMaxScaler` ile ölçekleme.
- `NearestNeighbors` (kosinüs benzerliği) ile her müşteri için davranışsal olarak en yakın 10 komşuyu bulur.
- Seçilen müşterinin komşuları arasındaki churn oranından bir risk skoru türetir.
- Kural tabanlı bir öneri motoru (`generate_banker_actions`) risk skoru, işlem sıklığı, inaktiflik süresi, gelir/limit oranı ve şikayet sıklığı gibi sinyallere göre bankacıya somut aksiyon önerileri üretir (ör. "limit artışı teklif edin", "önleyici arama yapın").

### Modül 3 — İstatistiksel Analiz

- Cinsiyete göre harcama farkını test etmek için normallik testi (Shapiro-Wilk) sonrasına göre otomatik olarak parametrik (T-Test) ya da non-parametrik (Mann-Whitney U) test seçimi yapar.
- `KMeans` ile toplam harcama ve işlem adedine göre 4 müşteri segmenti oluşturur (Bronze / Silver / Gold / Diamond) ve segmentlerin churn oranlarını karşılaştırır.
- Segment dağılımı, cinsiyete göre harcama (boxplot, violin plot, %95 güven aralıklı bar grafiği), segment bazlı churn oranı ve harcama histogramı olmak üzere altı farklı görselleştirme sekmesi sunar.

## Kullanılan Teknolojiler

- **Python 3**
- **Streamlit** — çok modüllü web arayüzü
- **pandas / NumPy** — veri işleme
- **scikit-learn** — `RandomForestClassifier`, `KMeans`, `NearestNeighbors`, `MinMaxScaler`, `LabelEncoder`
- **SciPy** — `shapiro`, `ttest_ind`, `mannwhitneyu` (hipotez testleri)
- **Matplotlib / Seaborn** — statik görselleştirmeler (Modül 3)
- **Plotly** — interaktif grafikler ve gauge göstergesi (Modül 1)
- **pyodbc** — MS SQL Server bağlantısı (yalnızca Modül 1 için gereklidir)

## Proje Yapısı

```
MiuulBank-Merkezi-Operasyon-Sistemi/
├── app.py                                   # Ana Streamlit uygulaması (3 modül)
├── data_prep.py                             # BankChurners.csv temizleme betiği
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── crm_measurement_analysis.ipynb       # Keşifsel analiz / istatistiksel testlerin notebook hâli
├── LICENSE
└── README.md
```

## Kurulum ve Çalıştırma

```bash
git clone https://github.com/fatihakcay23/MiuulBank-Merkezi-Operasyon-Sistemi.git
cd MiuulBank-Merkezi-Operasyon-Sistemi

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

1. [Kaggle üzerinden](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) `BankChurners.csv` dosyasını indirin ve proje klasörüne kopyalayın.
2. (Opsiyonel, yalnızca Modül 1 için) `data_prep.py` ile temizlenmiş veriyi üretip bir SQL Server veritabanına `BankChurners_Cleaned` adıyla yükleyin:

   ```bash
   python data_prep.py
   ```

   SQL Server bağlantı bilgilerini ortam değişkenleriyle belirtebilirsiniz:

   ```bash
   export MIUUL_SQL_SERVER="localhost\SQLEXPRESS"
   export MIUUL_SQL_DATABASE="miuulProje"
   ```

3. Uygulamayı başlatın:

   ```bash
   streamlit run app.py
   ```

Modül 2 ve Modül 3, `BankChurners.csv` proje klasöründe bulunduğu sürece SQL Server kurulumu gerektirmeden doğrudan çalışır. `MIUUL_DATA_PATH` ortam değişkeni ile farklı bir dosya konumu belirtilebilir.

## Bilinen Sınırlamalar

- Modül 1, çalışması için yerel/erişilebilir bir MS SQL Server örneği ve `ODBC Driver 17 for SQL Server` kurulumu gerektirir; bu bağımlılık olmadan yalnızca Modül 2 ve 3 kullanılabilir.
- Analizler ve model, tek bir zaman kesitindeki statik veriye dayanır; canlı/akan veri güncellemesi içermez.
- `RandomForestClassifier` ve `KMeans` hiperparametreleri sabittir; sistematik bir hiperparametre optimizasyonu yapılmamıştır.
- Öneri motorundaki (`generate_banker_actions`) eşik değerleri (ör. risk skoru %20) veri odaklı değil, sezgisel olarak belirlenmiştir.

## Geliştirme Fikirleri

- Sabit RandomForest/KMeans parametreleri yerine çapraz doğrulama ile hiperparametre optimizasyonu
- Model performansının ROC-AUC ve karışıklık matrisi ile genişletilmiş şekilde raporlanması
- SQL Server bağımlılığını ortadan kaldırmak için tüm modüllerin SQLite/CSV ile de çalışabilecek şekilde soyutlanması
- Öneri motorunun kural tabanlı yapıdan öğrenilmiş bir sıralama/skorlama modeline taşınması

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.
