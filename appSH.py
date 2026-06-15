import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# =========================
# INISIALISASI SESSION STATE
# =========================

if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'classification_proba' not in st.session_state:
    st.session_state.classification_proba = None
if 'regression_history' not in st.session_state:
    st.session_state.regression_history = []
if 'regression_result' not in st.session_state:
    st.session_state.regression_result = None

# =========================
# KONSTANTA DAN KONFIGURASI
# =========================

CLASSIFICATION_FEATURES = [
    "blood_pressure", "age", "gender", "bmi", "exercise_level",
    "smoking", "alcohol", "cholesterol", "glucose", "fatigue",
    "chest_pain", "dizziness", "diabetes", "stroke"
]

REGRESSION_FEATURES = [
    "age", "gender", "bmi", "exercise_level",
    "smoking", "alcohol", "cholesterol", "glucose", "fatigue",
    "chest_pain", "dizziness", "diabetes", "stroke", "heart_disease"
]

FEATURE_LABELS = {
    "blood_pressure": "Tekanan Darah (mmHg)",
    "age": "Usia (tahun)",
    "gender": "Jenis Kelamin",
    "bmi": "BMI",
    "exercise_level": "Tingkat Olahraga",
    "smoking": "Merokok",
    "alcohol": "Alkohol",
    "cholesterol": "Kolesterol (mg/dL)",
    "glucose": "Gula Darah (mg/dL)",
    "fatigue": "Kelelahan",
    "chest_pain": "Nyeri Dada",
    "dizziness": "Pusing",
    "diabetes": "Diabetes",
    "stroke": "Stroke",
    "heart_disease": "Penyakit Jantung"
}

BINARY_FEATURES = {
    "gender": {0: "Wanita", 1: "Pria"},
    "smoking": {0: "Tidak Merokok", 1: "Merokok"},
    "alcohol": {0: "Tidak Minum", 1: "Minum"},
    "fatigue": {0: "Tidak Kelelahan", 1: "Kelelahan"},
    "chest_pain": {0: "Tidak Nyeri Dada", 1: "Nyeri Dada"},
    "dizziness": {0: "Tidak Pusing", 1: "Pusing"},
    "diabetes": {0: "Tidak Diabetes", 1: "Diabetes"},
    "stroke": {0: "Tidak Stroke", 1: "Stroke"},
    "heart_disease": {0: "Tidak Ada", 1: "Ada"}
}

EXERCISE_LEVEL_OPTIONS = {
    0: "0 - Tidak pernah",
    1: "1 - Ringan (1-2x/minggu)",
    2: "2 - Teratur (3+x/minggu)"
}

CATEGORICAL_COLUMNS = [
    'gender', 'smoking', 'alcohol', 'fatigue', 
    'chest_pain', 'dizziness', 'diabetes', 'stroke', 
    'exercise_level', 'heart_disease'
]

NUMERIC_COLUMNS = [
    'blood_pressure', 'age', 'bmi', 'cholesterol', 'glucose'
]

# =========================
# FUNGSI UTILITY
# =========================

def find_dataset_file():
    all_files = []
    for pattern in ['*.csv', '*.xlsx', '*.xls']:
        all_files.extend(glob.glob(pattern))
    
    for file in all_files:
        if 'balanced' in file.lower():
            return file
    
    for file in all_files:
        file_lower = file.lower()
        if 'heart' in file_lower or 'disease' in file_lower or 'smart_healthcare' in file_lower:
            return file
    
    csv_files = glob.glob('*.csv')
    if csv_files:
        return csv_files[0]
    
    return None

def get_classification_models():
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)
    all_models = []
    for ext in ['*.joblib', '*.pkl']:
        all_models.extend(glob.glob(os.path.join(model_folder, ext)))
    
    classification_models = []
    for model_path in all_models:
        model_name = os.path.basename(model_path).lower()
        if any(k in model_name for k in ['classif', 'clf', 'heart', 'disease']):
            classification_models.append(model_path)
        elif not any(k in model_name for k in ['reg', 'blood']):
            classification_models.append(model_path)
    return classification_models

def get_regression_models():
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)
    all_models = []
    for ext in ['*.joblib', '*.pkl']:
        all_models.extend(glob.glob(os.path.join(model_folder, ext)))
    
    regression_models = []
    for model_path in all_models:
        model_name = os.path.basename(model_path).lower()
        if any(k in model_name for k in ['reg', 'blood']):
            regression_models.append(model_path)
    return regression_models

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

@st.cache_data
def load_dataset():
    dataset_file = find_dataset_file()
    if dataset_file is None:
        st.error("❌ Tidak ada file CSV/Excel yang ditemukan!")
        return None
    try:
        if dataset_file.endswith('.csv'):
            df = pd.read_csv(dataset_file)
        else:
            df = pd.read_excel(dataset_file)
        df = df.loc[:, ~df.columns.duplicated()]
        
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        st.success(f"✅ Dataset ditemukan: {dataset_file}")
        st.info(f"📊 {df.shape[0]} baris data, {df.shape[1]} kolom")
        
        with st.expander("📋 Kolom yang tersedia dalam dataset"):
            st.write(list(df.columns))
        
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def read_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df = df.loc[:, ~df.columns.duplicated()]
    
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df

def validate_classification_columns(df):
    return [col for col in CLASSIFICATION_FEATURES if col not in df.columns]

def validate_regression_columns(df):
    return [col for col in REGRESSION_FEATURES if col not in df.columns]

def prepare_classification_input(df):
    input_df = df[CLASSIFICATION_FEATURES].copy()
    for col in CLASSIFICATION_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
    return input_df

def prepare_regression_input(df):
    input_df = df[REGRESSION_FEATURES].copy()
    for col in REGRESSION_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
    return input_df

# =========================
# PENJELASAN DATASET
# =========================

def show_dataset_explanation(df):
    if df is None:
        st.warning("Dataset tidak tersedia.")
        return
    
    st.header("📖 Penjelasan Dataset")
    with st.expander("📋 Tentang Dataset", expanded=True):
        st.write(f"Dataset berisi **{df.shape[0]}** data pasien dengan **{df.shape[1]}** kolom.")
        if 'heart_disease' in df.columns:
            target_counts = df['heart_disease'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Tidak Ada Penyakit Jantung", target_counts.get(0, 0))
            with col2:
                st.metric("⚠️ Penyakit Jantung", target_counts.get(1, 0))
    
    with st.expander("🔍 Penjelasan Fitur", expanded=True):
        st.markdown("""
        | Fitur | Keterangan | Nilai |
        |-------|------------|-------|
        | **Tekanan Darah** | Tekanan darah sistolik (mmHg) | 90-120 |
        | **Usia** | Usia pasien (tahun) | 18-100 |
        | **Jenis Kelamin** | 0 = Wanita, 1 = Pria | 0/1 |
        | **BMI** | Indeks Massa Tubuh | 18.5-24.9 |
        | **Tingkat Olahraga** | 0=Tidak pernah, 1=Ringan, 2=Teratur | 0/1/2 |
        | **Merokok** | Kebiasaan merokok | 0=Tidak, 1=Ya |
        | **Alkohol** | Konsumsi alkohol | 0=Tidak, 1=Ya |
        | **Kolesterol** | Kadar kolesterol (mg/dL) | <200 |
        | **Gula Darah** | Kadar glukosa (mg/dL) | 70-100 |
        | **Kelelahan** | Adanya fatigue | 0=Tidak, 1=Ya |
        | **Nyeri Dada** | Adanya chest pain | 0=Tidak, 1=Ya |
        | **Pusing** | Adanya dizziness | 0=Tidak, 1=Ya |
        | **Diabetes** | Riwayat diabetes | 0=Tidak, 1=Ya |
        | **Stroke** | Riwayat stroke | 0=Tidak, 1=Ya |
        | **Penyakit Jantung** | Target klasifikasi | 0=Tidak, 1=Ya |
        """)
    
    with st.expander("📊 Sample Data"):
        st.dataframe(df.head(10), width='stretch')

# =========================
# VISUALISASI DATA - DIPERBAIKI
# =========================

def show_data_visualization(df):
    if df is None:
        st.warning("Dataset tidak tersedia.")
        return
    
    df = df.loc[:, ~df.columns.duplicated()]
    st.header("📊 Visualisasi Dataset")
    
    all_cols = df.columns.tolist()
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in all_cols]
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in all_cols]
    
    st.info(f"📌 **Fitur Numerik ({len(numeric_cols)}):** {', '.join([FEATURE_LABELS.get(c, c) for c in numeric_cols])}")
    st.info(f"📌 **Fitur Kategorikal ({len(categorical_cols)}):** {', '.join([FEATURE_LABELS.get(c, c) for c in categorical_cols])}")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Fitur", "🎯 Analisis Target", "📊 Korelasi"])
    
    with tab1:
        sub_tab1, sub_tab2 = st.tabs(["🔢 Distribusi Numerik", "🏷️ Distribusi Kategorikal"])
        
        with sub_tab1:
            st.subheader("Distribusi Fitur Numerik")
            if numeric_cols:
                selected_num = st.selectbox("Pilih fitur numerik", numeric_cols, format_func=lambda x: FEATURE_LABELS.get(x, x), key="num_feature")
                
                fig_hist = px.histogram(df, x=selected_num, title=f"Histogram - {FEATURE_LABELS.get(selected_num, selected_num)}", color_discrete_sequence=['#2E86AB'], nbins=30, marginal='box')
                fig_hist.update_layout(height=450)
                st.plotly_chart(fig_hist, width='stretch')
                
                fig_box = px.box(df, y=selected_num, title=f"Box Plot - {FEATURE_LABELS.get(selected_num, selected_num)}", color_discrete_sequence=['#A23B72'])
                fig_box.update_layout(height=450)
                st.plotly_chart(fig_box, width='stretch')
                
                fig_violin = px.violin(df, y=selected_num, title=f"Violin Plot - {FEATURE_LABELS.get(selected_num, selected_num)}", box=True, color_discrete_sequence=['#F18F01'])
                fig_violin.update_layout(height=450)
                st.plotly_chart(fig_violin, width='stretch')
            else:
                st.info("Tidak ada fitur numerik yang tersedia.")
        
        with sub_tab2:
            st.subheader("Distribusi Fitur Kategorikal")
            if categorical_cols:
                selected_cat = st.selectbox("Pilih fitur kategorikal", categorical_cols, format_func=lambda x: FEATURE_LABELS.get(x, x), key="cat_feature")
                data_cat = df[selected_cat].dropna()
                if len(data_cat) > 0:
                    counts = data_cat.value_counts().sort_index().reset_index()
                    counts.columns = ['value', 'count']
                    if selected_cat == 'exercise_level':
                        counts['label'] = counts['value'].map(EXERCISE_LEVEL_OPTIONS)
                    else:
                        counts['label'] = counts['value'].map(BINARY_FEATURES.get(selected_cat, {0: "Tidak", 1: "Ya"}))
                    counts['label'] = counts['label'].fillna(counts['value'].astype(str))
                    total = counts['count'].sum()
                    counts['percentage'] = (counts['count'] / total * 100).round(1)
                    
                    fig_bar = px.bar(counts, x='count', y='label', orientation='h', title=f"Distribusi {FEATURE_LABELS.get(selected_cat, selected_cat)}", color_discrete_sequence=['#F18F01'], text='count')
                    fig_bar.update_traces(textposition='outside', textfont_size=14)
                    fig_bar.update_layout(height=350, xaxis_title="Jumlah Pasien", yaxis_title=None)
                    st.plotly_chart(fig_bar, width='stretch')
                    
                    fig_donut = px.pie(counts, values='count', names='label', title=f"Proporsi {FEATURE_LABELS.get(selected_cat, selected_cat)}", color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
                    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                    fig_donut.update_layout(height=400)
                    st.plotly_chart(fig_donut, width='stretch')
                    
                    st.subheader(f"📋 Tabel Distribusi {FEATURE_LABELS.get(selected_cat, selected_cat)}")
                    table_data = []
                    for _, row in counts.iterrows():
                        table_data.append({'Kategori': row['label'], 'Nilai': row['value'], 'Jumlah Pasien': int(row['count']), 'Persentase (%)': row['percentage']})
                    table_data.append({'Kategori': 'TOTAL', 'Nilai': '-', 'Jumlah Pasien': int(total), 'Persentase (%)': 100.0})
                    st.dataframe(pd.DataFrame(table_data), width='stretch', hide_index=True)
                    
                    if len(counts) == 2:
                        count_1 = counts[counts['value'] == 1]['count'].values[0] if len(counts[counts['value'] == 1]) > 0 else 0
                        count_0 = counts[counts['value'] == 0]['count'].values[0] if len(counts[counts['value'] == 0]) > 0 else 0
                        col1, col2 = st.columns(2)
                        with col1: st.info(f"📌 **{FEATURE_LABELS.get(selected_cat, selected_cat)} = YA (1)**\nJumlah: **{int(count_1)}** pasien ({count_1/total*100:.1f}%)")
                        with col2: st.info(f"📌 **{FEATURE_LABELS.get(selected_cat, selected_cat)} = TIDAK (0)**\nJumlah: **{int(count_0)}** pasien ({count_0/total*100:.1f}%)")
            else:
                st.info("Tidak ada fitur kategorikal yang tersedia.")
    
    with tab2:
        if 'heart_disease' in df.columns:
            st.subheader("Analisis Target (Penyakit Jantung)")
            
            col1, col2 = st.columns(2)
            with col1:
                target_counts = df['heart_disease'].value_counts().reset_index()
                target_counts.columns = ['heart_disease', 'count']
                target_counts['heart_disease'] = target_counts['heart_disease'].map({0: 'Tidak Ada', 1: 'Ada'})
                fig_pie = px.pie(target_counts, values='count', names='heart_disease', title="Proporsi Penyakit Jantung", color_discrete_sequence=['#2E86AB', '#D64550'], hole=0.4)
                fig_pie.update_layout(height=450)
                st.plotly_chart(fig_pie, width='stretch')
            
            with col2:
                compare_options = numeric_cols + categorical_cols
                if compare_options:
                    selected_compare = st.selectbox("Pilih fitur untuk dibandingkan", compare_options, format_func=lambda x: FEATURE_LABELS.get(x, x), key="compare_feature")
                    if selected_compare in numeric_cols:
                        fig_box = px.box(df, x='heart_disease', y=selected_compare, title=f"{FEATURE_LABELS.get(selected_compare, selected_compare)} vs Penyakit Jantung", color='heart_disease', color_discrete_map={0: '#2E86AB', 1: '#D64550'})
                        fig_box.update_layout(height=450)
                        st.plotly_chart(fig_box, width='stretch')
                    else:
                        cross_tab = pd.crosstab(df[selected_compare], df['heart_disease'])
                        cross_tab.columns = ['Tidak Ada', 'Ada']
                        if selected_compare == 'exercise_level':
                            cross_tab.index = cross_tab.index.map(EXERCISE_LEVEL_OPTIONS)
                        else:
                            cross_tab.index = cross_tab.index.map(BINARY_FEATURES.get(selected_compare, {0: "Tidak", 1: "Ya"}))
                        plot_df = cross_tab.reset_index().melt(id_vars='index', var_name='heart_disease', value_name='count')
                        fig_bar = px.bar(plot_df, x='index', y='count', color='heart_disease', title=f"{FEATURE_LABELS.get(selected_compare, selected_compare)} vs Penyakit Jantung", barmode='group', color_discrete_map={'Tidak Ada': '#2E86AB', 'Ada': '#D64550'}, text='count')
                        fig_bar.update_traces(textposition='outside')
                        fig_bar.update_layout(height=450, xaxis_title=None)
                        st.plotly_chart(fig_bar, width='stretch')
            
            if numeric_cols:
                st.subheader("Perbandingan Rata-rata Fitur Numerik")
                mean_df = df.groupby('heart_disease')[numeric_cols].mean().T
                mean_df.columns = ['Tidak Ada', 'Ada']
                mean_df['Perbedaan'] = mean_df['Ada'] - mean_df['Tidak Ada']
                mean_df = mean_df.sort_values('Perbedaan', ascending=False)
                diff_df = mean_df[['Perbedaan']].reset_index()
                diff_df.columns = ['Fitur', 'Perbedaan']
                diff_df['Fitur'] = diff_df['Fitur'].map(lambda x: FEATURE_LABELS.get(x, x))
                fig_diff = px.bar(diff_df, x='Perbedaan', y='Fitur', orientation='h', title="Perbedaan Rata-rata (Ada - Tidak Ada)", color='Perbedaan', color_continuous_scale='RdBu', text='Perbedaan')
                fig_diff.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig_diff.update_layout(height=max(400, len(diff_df) * 30))
                st.plotly_chart(fig_diff, width='stretch')
        else:
            st.info("Tidak ada kolom 'heart_disease' untuk analisis target.")
    
    with tab3:
        st.subheader("Matriks Korelasi Antar Fitur")
        if len(numeric_cols) >= 2:
            selected_corr = st.multiselect("Pilih fitur untuk matriks korelasi (minimal 2)", numeric_cols, default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols, format_func=lambda x: FEATURE_LABELS.get(x, x), key="corr_features")
            if len(selected_corr) >= 2:
                corr_matrix = df[selected_corr].corr()
                fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", title="Matriks Korelasi Antar Fitur", color_continuous_scale='RdBu', zmin=-1, zmax=1)
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, width='stretch')
        if 'heart_disease' in df.columns and len(numeric_cols) > 0:
            st.subheader("Korelasi dengan Penyakit Jantung")
            corr_list = []
            for col in numeric_cols:
                if col != 'heart_disease':
                    corr_val = df[col].corr(df['heart_disease'])
                    if not pd.isna(corr_val):
                        corr_list.append((col, corr_val))
            corr_list.sort(key=lambda x: x[1], reverse=True)
            if corr_list:
                corr_df = pd.DataFrame(corr_list, columns=['Fitur', 'Korelasi'])
                corr_df['Fitur'] = corr_df['Fitur'].map(lambda x: FEATURE_LABELS.get(x, x))
                fig_corr = px.bar(corr_df, x='Korelasi', y='Fitur', orientation='h', title="Korelasi dengan Penyakit Jantung", color='Korelasi', color_continuous_scale='RdBu', text='Korelasi')
                fig_corr.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig_corr.update_layout(height=max(400, len(corr_df) * 30))
                st.plotly_chart(fig_corr, width='stretch')

# =========================
# KLASIFIKASI (HANYA FUNGSI PENTING - SESUAIKAN DENGAN KODE SEBELUMNYA)
# =========================

def train_classification_model(df):
    X = df[CLASSIFICATION_FEATURES]
    y = df['heart_disease']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def show_classification_manual(model, df_full):
    st.header("🎯 Klasifikasi - Prediksi Manual")
    st.write("Masukkan data pasien untuk memprediksi apakah memiliki penyakit jantung.")
    
    model_type = type(model).__name__
    st.caption(f"📌 Model yang digunakan: **{model_type}**")
    
    input_data = {}
    
    st.markdown("### 📊 Data Dasar")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["blood_pressure"] = st.number_input("Tekanan Darah (mmHg)", value=120.0, min_value=50.0, max_value=250.0, key="cls_bp")
    with col2:
        input_data["age"] = st.number_input("Usia (tahun)", value=50.0, min_value=18.0, max_value=100.0, key="cls_age")
    with col3:
        input_data["gender"] = 1 if st.selectbox("Jenis Kelamin", ["Wanita", "Pria"], key="cls_gender") == "Pria" else 0
    
    st.markdown("### ⚖️ Gaya Hidup")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["bmi"] = st.number_input("BMI", value=24.0, min_value=10.0, max_value=50.0, key="cls_bmi")
    with col2:
        input_data["exercise_level"] = st.selectbox("Tingkat Olahraga", [0, 1, 2], format_func=lambda x: EXERCISE_LEVEL_OPTIONS[x], key="cls_ex")
    with col3:
        input_data["smoking"] = 1 if st.selectbox("Merokok", ["Tidak", "Ya"], key="cls_smoke") == "Ya" else 0
    
    st.markdown("### 🩸 Laboratorium")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["cholesterol"] = st.number_input("Kolesterol (mg/dL)", value=200.0, min_value=100.0, max_value=400.0, key="cls_chol")
    with col2:
        input_data["glucose"] = st.number_input("Gula Darah (mg/dL)", value=100.0, min_value=50.0, max_value=300.0, key="cls_glu")
    with col3:
        input_data["alcohol"] = 1 if st.selectbox("Alkohol", ["Tidak", "Ya"], key="cls_alcohol") == "Ya" else 0
    
    st.markdown("### 🤒 Gejala")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["fatigue"] = 1 if st.selectbox("Kelelahan", ["Tidak", "Ya"], key="cls_fat") == "Ya" else 0
    with col2:
        input_data["chest_pain"] = 1 if st.selectbox("Nyeri Dada", ["Tidak", "Ya"], key="cls_chest") == "Ya" else 0
    with col3:
        input_data["dizziness"] = 1 if st.selectbox("Pusing", ["Tidak", "Ya"], key="cls_diz") == "Ya" else 0
    
    st.markdown("### 📋 Riwayat")
    col1, col2 = st.columns(2)
    with col1:
        input_data["diabetes"] = 1 if st.selectbox("Diabetes", ["Tidak", "Ya"], key="cls_diab") == "Ya" else 0
    with col2:
        input_data["stroke"] = 1 if st.selectbox("Stroke", ["Tidak", "Ya"], key="cls_str") == "Ya" else 0
    
    if st.button("🔍 Prediksi", type="primary", width='stretch'):
        try:
            input_df = pd.DataFrame([input_data])[CLASSIFICATION_FEATURES]
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            st.session_state.classification_result = prediction
            st.session_state.classification_proba = proba
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.session_state.classification_result is not None:
        prediction = st.session_state.classification_result
        proba = st.session_state.classification_proba
        
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("⚠️ **Penyakit jantung terdeteksi**")
            else:
                st.success("✅ **Tidak ada penyakit jantung**")
        with col2:
            st.metric("Probabilitas Penyakit Jantung", f"{proba[1]:.2%}")
        
        st.progress(proba[1])
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba[1] * 100,
            title={'text': "Tingkat Risiko (%)", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#D64550" if proba[1] > 0.5 else "#2E86AB"},
                'steps': [
                    {'range': [0, 30], 'color': '#CCFFCC'},
                    {'range': [30, 60], 'color': '#FFFFCC'},
                    {'range': [60, 100], 'color': '#FFCCCC'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': proba[1] * 100}
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, width='stretch')

def show_classification_csv(model):
    st.header("📁 Klasifikasi - Upload File")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            df = read_uploaded_file(uploaded_file)
            st.dataframe(df.head(), width='stretch')
            missing = validate_classification_columns(df)
            if missing:
                st.error(f"Kolom hilang: {missing}")
            elif st.button("Prediksi", type="primary", width='stretch'):
                input_df = prepare_classification_input(df)
                predictions = model.predict(input_df)
                result_df = df.copy()
                result_df["prediction"] = predictions
                result_df["status"] = ["✅ Sehat" if p == 0 else "⚠️ Berisiko" for p in predictions]
                st.subheader("Hasil Prediksi")
                st.dataframe(result_df, width='stretch')
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Hasil", csv_data, "hasil_prediksi.csv")
        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# REGRESI
# =========================

def train_regression_model(df):
    X = df[REGRESSION_FEATURES]
    y = df['blood_pressure']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def show_regression_manual(model):
    st.header("📈 Regresi - Prediksi Tekanan Darah")
    st.write("Masukkan data pasien untuk memprediksi nilai Tekanan Darah.")
    st.info("💡 Tekanan Darah adalah TARGET yang akan diprediksi")
    
    input_data = {}
    
    st.markdown("### 📊 Data Dasar")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["age"] = st.number_input("Usia (tahun)", value=50.0, min_value=18.0, max_value=100.0, key="reg_age")
    with col2:
        input_data["gender"] = 1 if st.selectbox("Jenis Kelamin", ["Wanita", "Pria"], key="reg_gender") == "Pria" else 0
    with col3:
        input_data["bmi"] = st.number_input("BMI", value=24.0, min_value=10.0, max_value=50.0, key="reg_bmi")
    
    st.markdown("### 🏃 Gaya Hidup")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["exercise_level"] = st.selectbox("Tingkat Olahraga", [0, 1, 2], format_func=lambda x: EXERCISE_LEVEL_OPTIONS[x], key="reg_ex")
    with col2:
        input_data["smoking"] = 1 if st.selectbox("Merokok", ["Tidak", "Ya"], key="reg_smoke") == "Ya" else 0
    with col3:
        input_data["alcohol"] = 1 if st.selectbox("Alkohol", ["Tidak", "Ya"], key="reg_alcohol") == "Ya" else 0
    
    st.markdown("### 🩸 Laboratorium")
    col1, col2 = st.columns(2)
    with col1:
        input_data["cholesterol"] = st.number_input("Kolesterol (mg/dL)", value=200.0, min_value=100.0, max_value=400.0, key="reg_chol")
    with col2:
        input_data["glucose"] = st.number_input("Gula Darah (mg/dL)", value=100.0, min_value=50.0, max_value=300.0, key="reg_glu")
    
    st.markdown("### 🤒 Gejala")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["fatigue"] = 1 if st.selectbox("Kelelahan", ["Tidak", "Ya"], key="reg_fat") == "Ya" else 0
    with col2:
        input_data["chest_pain"] = 1 if st.selectbox("Nyeri Dada", ["Tidak", "Ya"], key="reg_chest") == "Ya" else 0
    with col3:
        input_data["dizziness"] = 1 if st.selectbox("Pusing", ["Tidak", "Ya"], key="reg_diz") == "Ya" else 0
    
    st.markdown("### 📋 Riwayat")
    col1, col2, col3 = st.columns(3)
    with col1:
        input_data["diabetes"] = 1 if st.selectbox("Diabetes", ["Tidak", "Ya"], key="reg_diab") == "Ya" else 0
    with col2:
        input_data["stroke"] = 1 if st.selectbox("Stroke", ["Tidak", "Ya"], key="reg_str") == "Ya" else 0
    with col3:
        input_data["heart_disease"] = 1 if st.selectbox("Penyakit Jantung", ["Tidak", "Ya"], key="reg_hd") == "Ya" else 0
    
    if st.button("📊 Prediksi Tekanan Darah", type="primary", width='stretch'):
        try:
            input_df = pd.DataFrame([input_data])[REGRESSION_FEATURES]
            prediction = model.predict(input_df)[0]
            st.session_state.regression_result = prediction
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.session_state.regression_result is not None:
        prediction = st.session_state.regression_result
        
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediksi Tekanan Darah", f"{prediction:.1f} mmHg")
        with col2:
            st.info("Referensi: Normal 90-120 mmHg")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            title={'text': "Tekanan Darah (mmHg)", 'font': {'size': 20}},
            delta={'reference': 120, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 200], 'tickwidth': 1},
                'bar': {'color': "#2E86AB"},
                'steps': [
                    {'range': [0, 90], 'color': '#FFCCCC'},
                    {'range': [90, 120], 'color': '#CCFFCC'},
                    {'range': [120, 140], 'color': '#FFFFCC'},
                    {'range': [140, 200], 'color': '#FFCCCC'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prediction}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width='stretch')
        
        if prediction < 90:
            st.warning("⚠️ Hipotensi (Tekanan Darah Rendah)")
        elif prediction <= 120:
            st.success("✅ Tekanan Darah Normal")
        elif prediction <= 140:
            st.warning("⚠️ Prehipertensi")
        else:
            st.error("🔴 Hipertensi (Tekanan Darah Tinggi)")

def show_regression_csv(model):
    st.header("📁 Regresi - Upload File")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            df = read_uploaded_file(uploaded_file)
            st.dataframe(df.head(), width='stretch')
            missing = validate_regression_columns(df)
            if missing:
                st.error(f"Kolom hilang: {missing}")
            elif st.button("Prediksi", type="primary", width='stretch'):
                input_df = prepare_regression_input(df)
                predictions = model.predict(input_df)
                result_df = df.copy()
                result_df["predicted_blood_pressure"] = predictions
                st.subheader("Hasil Prediksi")
                st.dataframe(result_df, width='stretch')
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Hasil", csv_data, "hasil_regresi.csv")
        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# MAIN APP
# =========================

def main():
    st.title("❤️ Heart Disease Prediction App")
    st.markdown("---")
    
    df = load_dataset()
    if df is None:
        st.stop()
    
    st.sidebar.header("📌 Navigasi")
    menu = st.sidebar.radio("Pilih Menu", ["📖 Penjelasan Dataset", "📊 Visualisasi Data", "🎯 Klasifikasi", "📈 Regresi"])
    
    if menu == "📖 Penjelasan Dataset":
        show_dataset_explanation(df)
    
    elif menu == "📊 Visualisasi Data":
        show_data_visualization(df)
    
    elif menu == "🎯 Klasifikasi":
        st.header("🎯 Klasifikasi Penyakit Jantung")
        if 'heart_disease' not in df.columns:
            st.error("Dataset tidak memiliki kolom 'heart_disease'!")
            return
        
        classification_models = get_classification_models()
        if classification_models:
            st.sidebar.subheader("📦 Model Klasifikasi Tersedia")
            selected_model = st.sidebar.selectbox("Pilih model", classification_models, format_func=lambda x: os.path.basename(x))
            model = load_model(selected_model)
            st.sidebar.success(f"✅ Menggunakan: {os.path.basename(selected_model)}")
        else:
            st.info("📌 Tidak ada model klasifikasi. Menggunakan model default (Random Forest).")
            model = train_classification_model(df)
            st.success("✅ Model default siap digunakan!")
        
        method = st.radio("Pilih metode input", ["✏️ Input Manual", "📁 Upload File"], horizontal=True)
        if method == "✏️ Input Manual":
            show_classification_manual(model, df)
        else:
            show_classification_csv(model)
    
    elif menu == "📈 Regresi":
        st.header("📈 Regresi - Prediksi Tekanan Darah")
        regression_models = get_regression_models()
        if regression_models:
            st.sidebar.subheader("📦 Model Regresi Tersedia")
            selected_model = st.sidebar.selectbox("Pilih model regresi", regression_models, format_func=lambda x: os.path.basename(x))
            model = load_model(selected_model)
            st.sidebar.success(f"✅ Menggunakan: {os.path.basename(selected_model)}")
        else:
            st.info("📌 Tidak ada model regresi. Menggunakan model default (Random Forest).")
            model = train_regression_model(df)
            st.success("✅ Model default siap digunakan!")
        
        method = st.radio("Pilih metode input", ["✏️ Input Manual", "📁 Upload File"], horizontal=True)
        if method == "✏️ Input Manual":
            show_regression_manual(model)
        else:
            show_regression_csv(model)

if __name__ == "__main__":
    main()