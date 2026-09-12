import streamlit as st
import pandas as pd
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    # pyodbc / ODBC sürücüsü kurulu değilse uygulamanın geneli yine de çalışabilsin;
    # sadece "SQL & AI Analytics" modülü kullanılamaz.
    PYODBC_AVAILABLE = False
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

# =============================================================================
# GLOBAL AYARLAR (Tüm Sayfalar İçin Ortak)
# =============================================================================
st.set_page_config(
    page_title="MiuulBank Merkezi Operasyon Sistemi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# MODÜL 1: SQL & RANDOM FOREST
# =============================================================================
def app_sql_analytics():
    # Bağlantı bilgileri ortam değişkenlerinden okunur; tanımlı değilse yerel varsayılanlar kullanılır.
    # Kendi SQL Server örneğinizi kullanmak için MIUUL_SQL_SERVER / MIUUL_SQL_DATABASE
    # ortam değişkenlerini ayarlamanız yeterlidir.
    SERVER_NAME = os.environ.get('MIUUL_SQL_SERVER', 'localhost\\SQLEXPRESS')
    DATABASE_NAME = os.environ.get('MIUUL_SQL_DATABASE', 'miuulProje')
    TABLE_NAME = 'BankChurners_Cleaned'

    def get_db_connection():
        """MS SQL Veritabanına güvenli bağlantı kurar."""
        if not PYODBC_AVAILABLE:
            st.error(
                "pyodbc / ODBC Driver kurulu değil. Bu modül için 'pip install pyodbc' "
                "ve Microsoft ODBC Driver 17 for SQL Server gereklidir."
            )
            st.stop()
        try:
            conn = pyodbc.connect(
                f'DRIVER={{ODBC Driver 17 for SQL Server}};'
                f'SERVER={SERVER_NAME};'
                f'DATABASE={DATABASE_NAME};'
                'Trusted_Connection=yes;'
            )
            return conn
        except Exception as e:
            st.error(f"❌ Veritabanı Bağlantı Hatası: {e}")
            st.stop()

    def run_query(query):
        """SQL sorgusu çalıştırır ve Pandas DataFrame döner."""
        conn = get_db_connection()
        if conn:
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        return None

    @st.cache_resource
    def train_model():
        query = f"SELECT * FROM {TABLE_NAME}"
        df = run_query(query)

        leakage_cols = [c for c in df.columns if 'Naive_Bayes' in c]
        ignore_cols = ['clientnum', 'attrition_flag', 'churn_label'] + leakage_cols
        X = df.drop([c for c in ignore_cols if c in df.columns], axis=1)
        y = df['churn_label']
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return model, X.columns, accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test,
                                                                                                               y_pred)

    # --- SAYFA TASARIMI ---
    st.markdown("## 🏦 MiuulBank SQL & AI Analytics")
    menu = st.radio("Alt İşlemler", ["Ana Sayfa (KPI)", "Müşteri Analizi (SQL)", "Yapay Zeka Tahmini"], horizontal=True)
    st.success("Veritabanı Bağlantısı: Aktif ✅")
    st.markdown("---")

    # --- MENÜ 1: ANA SAYFA (KPI) ---
    if menu == "Ana Sayfa (KPI)":
        st.subheader("📊 Yönetici Özeti")
        st.markdown("Veritabanından anlık çekilen banka performans metrikleri.")

        # KPI Sorgusu
        df_kpi = run_query(
            f"SELECT COUNT(*) as TotalCust, SUM(CASE WHEN churn_label = 1 THEN 1 ELSE 0 END) as ChurnCount, AVG(credit_limit) as AvgLimit, AVG(total_trans_amt) as AvgSpend FROM {TABLE_NAME}")

        col1, col2, col3, col4 = st.columns(4)
        total = df_kpi['TotalCust'][0]
        churn = df_kpi['ChurnCount'][0]
        col1.metric("Toplam Müşteri", f"{total:,}")
        col2.metric("Kayıp Oranı", f"%{(churn / total) * 100:.1f}", delta_color="inverse")
        col3.metric("Ort. Kredi Limiti", f"${df_kpi['AvgLimit'][0]:,.0f}")
        col4.metric("Ort. Harcama", f"${df_kpi['AvgSpend'][0]:,.0f}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Eğitim Seviyesine Göre Dağılım")
            df_edu = run_query(f"SELECT education_level, COUNT(*) as Count FROM {TABLE_NAME} GROUP BY education_level")
            st.plotly_chart(px.pie(df_edu, names='education_level', values='Count', hole=0.4), use_container_width=True)
        with c2:
            st.subheader("Kart Tipi Bazında Ort. Harcama")
            df_card = run_query(
                f"SELECT card_category, AVG(total_trans_amt) as AvgSpend FROM {TABLE_NAME} GROUP BY card_category")
            st.plotly_chart(px.bar(df_card, x='card_category', y='AvgSpend', color='card_category'),
                            use_container_width=True)

    # --- MENÜ 2: MÜŞTERİ ANALİZİ (SQL) ---
    elif menu == "Müşteri Analizi (SQL)":
        st.subheader("🔍 Stratejik Raporlama Merkezi")
        st.markdown("Veriye dayalı karar alma mekanizması için hazırlanmış SQL senaryoları.")

        report_type = st.radio("Rapor Kategorisi:",
                               ["🚨 Erken Uyarı & Risk", "💎 Satış & Büyüme Fırsatları", "📉 Operasyonel Analiz"],
                               horizontal=True)

        scenarios = {}

        if report_type == "🚨 Erken Uyarı & Risk":
            scenarios = {
                "Sessiz Kayıplar (Pasifleşen VIP'ler)": {
                    "desc": "Kredi limiti 10.000$ üstü olup son 3 aydır inaktif olan müşteriler. Terk etme riski çok yüksektir.",
                    "query": f"SELECT clientnum, customer_age, credit_limit, months_inactive_12_mon, total_trans_amt FROM {TABLE_NAME} WHERE credit_limit > 10000 AND months_inactive_12_mon >= 3 ORDER BY credit_limit DESC"
                },
                "Limitini Zorlayanlar (Borç Riski)": {
                    "desc": "Limit kullanım oranı %90'ın üzerinde olan ve henüz Churn olmamış müşteriler. Ödeme güçlüğü çekebilirler.",
                    "query": f"SELECT clientnum, credit_limit, total_revolving_bal, avg_utilization_ratio FROM {TABLE_NAME} WHERE avg_utilization_ratio > 0.90 AND churn_label = 0 ORDER BY total_revolving_bal DESC"
                },
                "Ani İşlem Kesintisi": {
                    "desc": "Geçmişte çok aktif olup (60+ işlem), son dönemde işlem sayısı %50'den fazla düşen sadık müşteriler.",
                    "query": f"SELECT clientnum, total_trans_ct, total_ct_chng_q4_q1, total_trans_amt FROM {TABLE_NAME} WHERE total_trans_ct > 60 AND total_ct_chng_q4_q1 < 0.5 ORDER BY total_ct_chng_q4_q1 ASC"
                },
                "Riskli Aktif Müşteriler": {
                    "desc": "Şu an bankayı terk etmiş (Churn=1) ancak geçmiş harcaması yüksek olan, geri kazanılması gereken müşteriler.",
                    "query": f"SELECT TOP 200 clientnum, customer_age, total_trans_amt, avg_utilization_ratio FROM {TABLE_NAME} WHERE churn_label = 1 ORDER BY total_trans_amt DESC"
                }
            }

        elif report_type == "💎 Satış & Büyüme Fırsatları":
            scenarios = {
                "Uyuyan Devler (Yüksek Gelir)": {
                    "desc": "Geliri 80K$+ olup kartını neredeyse hiç kullanmayanlar. Özel kampanya ile uyandırılmalılar.",
                    "query": f"SELECT clientnum, income_category, credit_limit, avg_utilization_ratio FROM {TABLE_NAME} WHERE (income_category = '$80K - $120K' OR income_category = '$120K +') AND avg_utilization_ratio < 0.10 ORDER BY credit_limit DESC"
                },
                "Limit Artış Adayları": {
                    "desc": "Limiti 5000$ altı ama limiti dolmak üzere olan sadık müşteriler. Limit artışı teklifi için uygundur.",
                    "query": f"SELECT clientnum, credit_limit, avg_utilization_ratio, total_trans_amt FROM {TABLE_NAME} WHERE credit_limit < 5000 AND avg_utilization_ratio > 0.70 AND churn_label = 0 ORDER BY avg_utilization_ratio DESC"
                },
                "Platinum Kart Adayları": {
                    "desc": "Blue kart sahibi olup 10.000$ üzeri harcama yapan müşteriler. Kart yükseltme (Upgrade) teklifi verilmeli.",
                    "query": f"SELECT clientnum, card_category, total_trans_amt, total_trans_ct FROM {TABLE_NAME} WHERE card_category = 'Blue' AND total_trans_amt > 10000 ORDER BY total_trans_amt DESC"
                },
                "Genç Profesyoneller": {
                    "desc": "30 yaş altı ve yüksek eğitimli (Doktora/Yüksek Lisans) potansiyel müşteriler.",
                    "query": f"SELECT clientnum, customer_age, education_level, income_category FROM {TABLE_NAME} WHERE customer_age < 30 AND (education_level = 'Graduate' OR education_level = 'Doctorate') ORDER BY customer_age ASC"
                }
            }

        elif report_type == "📉 Operasyonel Analiz":
            scenarios = {
                "Eğitim Seviyesi & Harcama": {
                    "desc": "Hangi eğitim seviyesindeki müşteriler bankaya daha çok kazandırıyor?",
                    "query": f"SELECT education_level, COUNT(*) as Müşteri, AVG(total_trans_amt) as Ort_Harcama FROM {TABLE_NAME} GROUP BY education_level ORDER BY Ort_Harcama DESC"
                },
                "En Sadık Müşteriler (4+ Yıl)": {
                    "desc": "48 aydan uzun süredir bankada olan ve hala aktif en değerli müşteriler.",
                    "query": f"SELECT TOP 500 clientnum, months_on_book, total_relationship_count, total_trans_amt FROM {TABLE_NAME} WHERE months_on_book >= 48 AND churn_label = 0 ORDER BY total_trans_amt DESC"
                },
                "Kart Karlılık Analizi": {
                    "desc": "Kart tiplerine göre (Blue, Silver vb.) toplam işlem hacmi karşılaştırması.",
                    "query": f"SELECT card_category, SUM(total_trans_amt) as Toplam_Hacim, AVG(avg_utilization_ratio) as Ort_Kullanım FROM {TABLE_NAME} GROUP BY card_category ORDER BY Toplam_Hacim DESC"
                }
            }

        description_placeholder = st.empty()
        selected_option = st.selectbox("Senaryo Seçiniz:", list(scenarios.keys()), key=f"sb_{report_type}")
        selected_data = scenarios[selected_option]
        description = selected_data["desc"]
        sql_query = selected_data["query"]

        description_placeholder.info(f"💡 **Senaryo Açıklaması:** {description}")

        with st.expander("🛠️ SQL Sorgusunu Görüntüle"):
            st.code(sql_query, language="sql")

        if st.button("Raporu Oluştur ve Getir", type="primary"):
            df_report = run_query(sql_query)
            if df_report is not None and not df_report.empty:
                st.success(f"✅ Toplam {len(df_report)} kayıt bulundu.")
                st.dataframe(df_report, use_container_width=True)
                csv = df_report.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Excel İndir", csv, "rapor.csv", "text/csv")
            else:
                st.warning("⚠️ Kayıt bulunamadı.")

    # --- MENÜ 3: YAPAY ZEKA TAHMİNİ ---
    elif menu == "Yapay Zeka Tahmini":
        st.subheader("🤖 Gelişmiş Risk Analizi & Karar Destek")
        st.markdown("---")

        model, model_cols, acc, prec, rec = train_model()

        col_main1, col_main2 = st.columns([1, 2])

        with col_main1:
            st.markdown("### 👤 Profil Simülasyonu")
            with st.form("prediction_form"):
                st.markdown("**1. Mevcut Durum**")
                age = st.slider("Yaş", 20, 80, 46)
                trans_amt = st.number_input("Yıllık Harcama ($)", 0, 30000, 4000)
                trans_ct = st.number_input("İşlem Adedi", 0, 150, 60)

                st.markdown("**2. Finansal Durum**")
                inactive = st.slider("İnaktif Olduğu Ay Sayısı", 0, 12, 2)
                revolving = st.number_input("Dönen Bakiye (Borç)", 0, 5000, 1000)
                limit = st.number_input("Kredi Limiti", 1000, 40000, 8000)

                submitted = st.form_submit_button("Risk Analizini Çalıştır", type="primary")

        with col_main2:
            if submitted:
                utilization = revolving / limit if limit > 0 else 0
                default_change = 0.0 if trans_ct == 0 else 0.7

                input_df = pd.DataFrame({
                    'customer_age': [age],
                    'total_trans_amt': [trans_amt],
                    'total_trans_ct': [trans_ct],
                    'months_inactive_12_mon': [inactive],
                    'total_revolving_bal': [revolving],
                    'credit_limit': [limit],
                    'avg_utilization_ratio': [utilization],
                    'total_ct_chng_q4_q1': [default_change],
                    'total_amt_chng_q4_q1': [default_change],
                    'avg_open_to_buy': [limit - revolving],
                    'dependent_count': [2],
                    'months_on_book': [36],
                    'total_relationship_count': [3],
                    'contacts_count_12_mon': [3 if inactive > 2 else 2]
                })

                input_encoded = pd.get_dummies(input_df)
                input_encoded = input_encoded.reindex(columns=model_cols, fill_value=0)

                prob_risk = model.predict_proba(input_encoded)[0][1]
                risk_score = int(prob_risk * 100)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("Tahmin Sonucu")
                    if risk_score > 50:
                        st.error(f"🚨 RİSKLİ MÜŞTERİ")
                        st.metric("Terk Etme İhtimali", f"%{risk_score}", delta="Yüksek Risk", delta_color="inverse")
                        if inactive > 3:
                            st.warning("⚠️ Müşteri uzun süredir pasif durumda.")
                        if trans_ct == 0:
                            st.warning("⚠️ Hiç işlem yapmıyor.")
                    else:
                        st.success(f"✅ SADIK MÜŞTERİ")
                        st.metric("Sadakat Skoru", f"%{100 - risk_score}", delta="Güvenli", delta_color="normal")

                with col2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=risk_score, title={'text': "Risk Endeksi"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkred" if risk_score > 50 else "green"}}
                    ))
                    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("---")

                st.header("📊 Model Performansı")
                tab1, tab2, tab3 = st.tabs(["🧠 Özellik Önemi", "📈 Gerçek Metrikler", "🔥 Korelasyonlar"])

                with tab1:
                    importances = model.feature_importances_
                    feature_imp_df = pd.DataFrame({'Feature': model_cols, 'Importance': importances}).sort_values(
                        by='Importance', ascending=False).head(10)
                    fig_imp = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                                     title="Model Kararını Etkileyen Faktörler")
                    st.plotly_chart(fig_imp, use_container_width=True)

                with tab2:
                    st.write("Modelin test verisi üzerindeki **gerçek** performans skorları:")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Doğruluk (Accuracy)", f"%{acc * 100:.1f}")
                    c2.metric("Kesinlik (Precision)", f"%{prec * 100:.1f}")
                    c3.metric("Duyarlılık (Recall)", f"%{rec * 100:.1f}")
                    st.info("Bu değerler veritabanındaki verilerle anlık hesaplanmıştır.")

                with tab3:
                    conn = get_db_connection()
                    if conn:
                        df_corr = pd.read_sql(f"SELECT TOP 1000 * FROM {TABLE_NAME}", conn)
                        numeric_df = df_corr.select_dtypes(include=[np.number])
                        fig_corr = px.imshow(numeric_df.corr(), text_auto=False, aspect="auto",
                                             title="Korelasyon Haritası")
                        st.plotly_chart(fig_corr, use_container_width=True)
                        conn.close()


# =============================================================================
# MODÜL 2: PROAKTİF BANKACI EKRANI
# =============================================================================
def app_proactive_banker():
    def grab_col_names(dataframe, cat_th=10, car_th=20):
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if
                       dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if
                       dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]
        return cat_cols, num_cols, cat_but_car

    def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    @st.cache_resource
    def prepare_system():
        paths = [
            os.environ.get("MIUUL_DATA_PATH", ""),
            "BankChurners.csv",
            "BankChurners_Cleaned.csv",
        ]
        paths = [p for p in paths if p]
        df = None
        for p in paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
        if df is None:
            return None, None, None, None

        df.columns = df.columns.str.lower()
        # Churn Label kopyayı engellemek için siliniyor ama Attrition_Flag kalıyor (Analiz için)
        cols_to_drop = [col for col in df.columns if 'naive_bayes' in col] + ['churn_label']
        raw_df = df.drop(columns=cols_to_drop, errors='ignore').copy()
        processed_df = raw_df.copy()

        cat_cols, num_cols, cat_but_car = grab_col_names(processed_df)

        for col in num_cols:
            if col != "clientnum":
                replace_with_thresholds(processed_df, col)

        le = LabelEncoder()
        binary_cols = [col for col in processed_df.columns if
                       processed_df[col].dtype == 'O' and processed_df[col].nunique() == 2]
        for col in binary_cols:
            processed_df[col] = le.fit_transform(processed_df[col])

        ohe_cols = [col for col in processed_df.columns if
                    processed_df[col].dtype == 'O' and 10 >= processed_df[col].nunique() > 2]
        processed_df = pd.get_dummies(processed_df, columns=ohe_cols, drop_first=True)

        scaler = MinMaxScaler()
        cols_to_exclude = ["clientnum", "attrition_flag"]
        sc_cols = [col for col in processed_df.columns if col not in cols_to_exclude]

        processed_df[sc_cols] = scaler.fit_transform(processed_df[sc_cols])

        if 'attrition_flag' in processed_df.columns:
            model_df = processed_df.drop(columns=['attrition_flag'])
        else:
            model_df = processed_df

        if 'clientnum' in model_df.columns:
            model_df.set_index('clientnum', inplace=True)

        model = NearestNeighbors(n_neighbors=10, metric='cosine')
        model.fit(model_df)

        if 'clientnum' in processed_df.columns:
            processed_df.set_index('clientnum', inplace=True)

        return raw_df, processed_df, model, model_df

    def generate_banker_actions(client_data, churn_risk_score):
        actions = []
        if churn_risk_score >= 20:  # Risk eşiğini biraz düşürdük ki daha hassas olsun
            actions.append(("🚨 RİSK UYARISI",
                            f"Benzer müşterilerin %{churn_risk_score:.0f}'u bankayı terk etmiş. Önleyici arama yapın."))
        if client_data['total_trans_ct'] < 40:
            actions.append(("📉 Kullanım Azalmış", "Kredi kartı kullanımı düşük seviyede. Bonus kampanyası sunun."))
        if client_data['months_inactive_12_mon'] >= 3:
            actions.append(("💤 Uyuyan Müşteri", "Son 3 aydır işlem yapmıyor. 'Sizi Özledik' iletişimi planlayın."))
        income_high = any(x in str(client_data['income_category']) for x in ['$80K', '$120K'])
        if (client_data['credit_limit'] < 5000) and income_high:
            actions.append(("💰 Limit Fırsatı", "Yüksek gelirli ancak limiti düşük. Limit artırımı teklif edin."))
        if client_data['contacts_count_12_mon'] > 4:
            actions.append(
                ("📞 Memnuniyet Riski", "Bankayla çok sık iletişime geçmiş. Bir sorunu çözülememiş olabilir."))
        if client_data['total_revolving_bal'] > 2000:
            actions.append(
                ("💸 Faiz Geliri/Riski", "Yüksek borç bakiyesi var. Ödeme güçlüğü çekip çekmediğini analiz edin."))
        if not actions:
            actions.append(
                ("✅ Her Şey Yolunda", "Müşteri stabil. İlişkiyi sıcak tutmak için özel gün kutlaması yapabilirsiniz."))
        return actions

    # --- UYGULAMA MANTIĞI ---
    try:
        raw_df, processed_df, model, model_df = prepare_system()
    except Exception as e:
        st.error(f"Sistem hazırlanırken hata oluştu: {e}")
        return

    if raw_df is None:
        st.error("Veri dosyası bulunamadı! Dosya yolunu kontrol ediniz.")
        return

    st.subheader("🛡️ Müşteri Risk ve Benzerlik Paneli (KNN)")
    st.info(
        "Yapay zeka, sadece müşteri davranışlarına (Harcama, Limit, Aktiflik vb.) bakarak en çok benzeyen diğer müşterileri bulur ve riski hesaplar.")

    col_sel, col_info = st.columns([1, 3])
    with col_sel:
        # Rastgelelik yerine sıralı liste veya arama yapılabilir ama şimdilik selectbox kalsın
        selected_client = st.selectbox("Müşteri ID Seçiniz:", raw_df['clientnum'].unique())

    client_row = raw_df[raw_df['clientnum'] == selected_client].iloc[0]

    with col_info:
        status = str(client_row['attrition_flag'])
        if "Attrited" in status:
            st.error(f"DURUM: KAYIP (CHURN) - {selected_client}")
        else:
            st.success(f"DURUM: MEVCUT - {selected_client}")

    st.markdown("---")

    try:
        # Seçilen müşteriyi bul
        query = model_df.loc[selected_client].values.reshape(1, -1)

        # En yakın komşuları bul (Kendisi dahil ilk 10)
        distances, indices = model.kneighbors(query)

        # Kendisini (ilk elemanı) listeden çıkar, geriye kalan 9 benzere bak
        neighbor_indices = indices.flatten()[1:]

        # Bu indekslerin gerçek Client ID'lerini al
        similar_ids = model_df.index[neighbor_indices]

        # Raw Data'dan bu ID'lere sahip olanları çek
        similar_clients_df = raw_df[raw_df['clientnum'].isin(similar_ids)].copy()

        # Risk Hesaplama: "Attrited Customer" olanları say
        # Büyük/küçük harf duyarlılığını kaldırıp string içinde arıyoruz
        churn_count = similar_clients_df['attrition_flag'].apply(
            lambda x: 1 if "attrited" in str(x).lower() else 0).sum()
        total_neighbors = len(similar_clients_df)

        if total_neighbors > 0:
            risk_score = (churn_count / total_neighbors) * 100
        else:
            risk_score = 0

    except Exception as e:
        st.error(f"Analiz hatası: {e}")
        risk_score = 0
        similar_clients_df = pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Müşteri Yaşı", client_row['customer_age'])
    c2.metric("Toplam Harcama", f"${client_row['total_trans_amt']:,.0f}")
    c3.metric("İnaktif Aylar", client_row['months_inactive_12_mon'])

    # Risk Rengi Ayarı
    risk_color = "normal"
    if risk_score > 0: risk_color = "inverse"  # 0'dan büyük herhangi bir risk dikkat çekmeli

    c4.metric("Tahmini Kayıp Riski", f"%{risk_score:.1f}", delta_color=risk_color,
              help=f"Bu müşteriye davranışsal olarak en çok benzeyen {total_neighbors} kişiden {churn_count} tanesi bankayı terk etmiş.")

    st.subheader("💡 Bankacı İçin Önerilen Aksiyonlar")
    actions = generate_banker_actions(client_row, risk_score)
    for title, desc in actions:
        with st.expander(f"{title}", expanded=True):
            st.write(desc)
            if title == "🚨 RİSK UYARISI" or title == "🚨 ACİL":
                st.button("Müşteriyi Şimdi Ara", key="call_btn")
            elif "Kampanya" in desc or "Teklif" in desc:
                st.button("Teklif Gönder", key=f"email_{title}")

    st.markdown("---")

    # --- BENZER MÜŞTERİ LİSTESİ ---
    st.subheader(f"🔍 Referans Alınan Benzer Müşteriler")
    st.write(
        f"Sistem, **{selected_client}** nolu müşteriye en çok benzeyen kişileri analiz etti. Risk puanı buradaki müşterilerin durumuna göre hesaplandı.")

    if not similar_clients_df.empty:
        display_cols = ['clientnum', 'attrition_flag', 'customer_age', 'total_trans_amt', 'credit_limit',
                        'months_inactive_12_mon']

        def highlight_churn(val):
            val_str = str(val).lower()
            if 'attrited' in val_str:
                return 'color: red; font-weight: bold'
            return 'color: green'

        st.dataframe(
            similar_clients_df[display_cols].style.map(highlight_churn, subset=['attrition_flag'])
            .format({"total_trans_amt": "${:,.0f}", "credit_limit": "${:,.0f}"}),
            use_container_width=True
        )
    else:
        st.info("Benzer müşteri bulunamadı.")

    # --- YENİ EKLENEN KISIM: SEÇİLİ MÜŞTERİ DETAYLARI ---
    st.markdown("---")
    st.subheader(f"📋 Seçili Müşteri Detay Kartı: {selected_client}")
    st.markdown("Müşterinin sistemdeki tüm güncel bilgileri aşağıdadır.")

    # Müşteri verisini daha okunaklı olması için Transpose (T) ediyoruz (Satır sütun yer değiştiriyor)
    client_detail_df = pd.DataFrame(client_row).T

    # Tüm kolonları gösterelim ama formatlayalım
    st.dataframe(
        client_detail_df.style.format(precision=2),
        use_container_width=True
    )
# =============================================================================
# MODÜL 3: BANKA MÜŞTERİ ANALİZİ
# =============================================================================
def app_customer_stats():
    # Pandas ayarları (Görünüm için)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    sns.set(style="whitegrid")

    st.subheader("📊 Banka Müşteri Kayıp (Churn) ve Segmentasyon Raporu")

    candidate_paths = [
        os.environ.get("MIUUL_DATA_PATH", ""),
        "BankChurners.csv",
        "BankChurners_Cleaned.csv",
    ]
    candidate_paths = [p for p in candidate_paths if p]

    @st.cache_data
    def load_data(paths):
        for path in paths:
            if os.path.exists(path):
                data = pd.read_csv(path)
                data = data.iloc[:, :-2]
                data["Churn"] = data["Attrition_Flag"].apply(lambda x: 1 if x == "Attrited Customer" else 0)
                return data, path
        return None, None

    df, used_path = load_data(candidate_paths)

    if df is not None:
        st.success(f"Veri Seti Yüklendi ({used_path})")
    else:
        st.error("HATA: BankChurners.csv bulunamadı! Dosyayı proje klasörüne ekleyin veya MIUUL_DATA_PATH ortam değişkenini ayarlayın.")
        st.stop()

    def ab_testing_gender_spend_web(dataframe):
        st.subheader("🛠️ 2. A/B Testi: Cinsiyete Göre Harcama Farklılığı")
        group_m = dataframe[dataframe["Gender"] == "M"]["Total_Trans_Amt"]
        group_f = dataframe[dataframe["Gender"] == "F"]["Total_Trans_Amt"]
        stat_m, p_m = shapiro(group_m)
        stat_f, p_f = shapiro(group_f)
        col1, col2 = st.columns(2)
        col1.metric("Erkek Normallik P-Value", f"{p_m:.4f}")
        col2.metric("Kadın Normallik P-Value", f"{p_f:.4f}")
        if (p_m > 0.05) and (p_f > 0.05):
            stat, p_value = ttest_ind(group_m, group_f, equal_var=True)
            test_type = "T-Test (Parametrik)"
        else:
            stat, p_value = mannwhitneyu(group_m, group_f)
            test_type = "Mann-Whitney U (Non-Parametrik)"
        st.info(f"Uygulanan Test: **{test_type}**")
        st.write(f"Test Sonucu P-Value: **{p_value:.4f}**")
        if p_value < 0.05:
            st.success("SONUÇ: H0 Reddedilir. İki grup arasında ANLAMLI bir fark vardır.")
        else:
            st.warning("SONUÇ: H0 Reddedilemez. Anlamlı bir fark yoktur.")

    ab_testing_gender_spend_web(df)

    def create_segments_kmeans_web(dataframe):
        st.subheader("🧩 3. K-Means Müşteri Segmentasyonu")
        rfm_df = dataframe[["Total_Trans_Amt", "Total_Trans_Ct"]]
        sc = MinMaxScaler((0, 1))
        rfm_scaled = sc.fit_transform(rfm_df)
        kmeans = KMeans(n_clusters=4, random_state=42)
        k_fit = kmeans.fit(rfm_scaled)
        dataframe["Segment_No"] = k_fit.labels_
        summary = dataframe.groupby("Segment_No")[["Total_Trans_Amt", "Total_Trans_Ct"]].mean()
        summary["Score"] = summary["Total_Trans_Amt"] + summary["Total_Trans_Ct"]
        summary = summary.sort_values("Score")
        label_map = {
            summary.index[0]: "Bronze (Düşük)",
            summary.index[1]: "Silver (Orta)",
            summary.index[2]: "Gold (Yüksek)",
            summary.index[3]: "Diamond (VIP)"
        }
        dataframe["Segment_Name"] = dataframe["Segment_No"].map(label_map)
        st.write("Segment Dağılımı:")
        st.write(dataframe["Segment_Name"].value_counts())
        return dataframe

    df = create_segments_kmeans_web(df)

    st.subheader("📈 4. Detaylı Görselleştirme")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Scatter Segment", "Boxplot Cinsiyet", "Churn Oranları", "Violin Plot", "Güven Aralığı", "Histogram"])

    with tab1:
        st.write("**Segment Dağılımı**")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="Total_Trans_Ct", y="Total_Trans_Amt", hue="Segment_Name", data=df, palette="viridis", s=60,
                        alpha=0.9, ax=ax1)
        st.pyplot(fig1)

    with tab2:
        st.write("**Cinsiyete Göre Harcama**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x="Gender", y="Total_Trans_Amt", data=df, palette="coolwarm", ax=ax2)
        st.pyplot(fig2)

    with tab3:
        st.write("**Segmentlerin Churn Oranları**")
        churn_rates = df.groupby("Segment_Name")["Churn"].mean().reset_index().sort_values("Churn", ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        barplot = sns.barplot(x="Segment_Name", y="Churn", data=churn_rates, palette="Reds_r", ax=ax3)
        ax3.set_ylim(0, 0.6)
        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.2%'), (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        st.pyplot(fig3)

    with tab4:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.violinplot(x="Gender", y="Total_Trans_Amt", data=df, palette="muted", inner="quartile", ax=ax4)
        st.pyplot(fig4)

    with tab5:
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        ax_bar = sns.barplot(x="Gender", y="Total_Trans_Amt", data=df, palette="pastel", errorbar=('ci', 95),
                             capsize=0.1, ax=ax5)
        st.pyplot(fig5)

    with tab6:
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x="Total_Trans_Amt", hue="Gender", kde=True, element="step", palette="dark", alpha=0.6,
                     ax=ax6)
        st.pyplot(fig6)


# =============================================================================
# ANA GİRİŞ EKRANI (LANDING PAGE)
# =============================================================================
def main():
    st.sidebar.title("🏦 BANKA PANELİ")
    st.sidebar.markdown("Hoşgeldiniz.")

    # Navigasyon Menüsü
    app_choice = st.sidebar.radio("Uygulama Seçiniz:",
                                  ["Ana Giriş",
                                   "1. SQL & AI Analytics",
                                   "2. Proaktif Bankacı (Risk)",
                                   "3. İstatistiksel Analiz"])

    st.sidebar.markdown("---")
    st.sidebar.info("Miuul Finansal Teknoloji Çözümleri v1.0")

    if app_choice == "Ana Giriş":
        st.title("Merkezi Operasyon Sistemi'ne Hoşgeldiniz 👋")
        st.markdown("### Lütfen yapmak istediğiniz işlemi seçin:")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.info("📊 **SQL & AI Analytics**")
            st.markdown("Veritabanına bağlanın, SQL raporları çekin ve Random Forest modeli ile tahminleme yapın.")

        with c2:
            st.warning("🛡️ **Proaktif Bankacı**")
            st.markdown("Müşteri davranışlarına göre benzerlik (KNN) analizi yapın ve aksiyon önerileri alın.")

        with c3:
            st.success("📈 **İstatistiksel Analiz**")
            st.markdown("A/B testleri, K-Means segmentasyonu ve detaylı grafiklerle veri keşfi yapın.")

        st.markdown("---")
        st.image(
            "https://images.unsplash.com/photo-1565514020179-0222d7b20399?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
            use_container_width=True, caption="Bankacılıkta Veri Gücü")

    elif app_choice == "1. SQL & AI Analytics":
        app_sql_analytics()

    elif app_choice == "2. Proaktif Bankacı (Risk)":
        app_proactive_banker()

    elif app_choice == "3. İstatistiksel Analiz":
        app_customer_stats()


if __name__ == "__main__":
    main()