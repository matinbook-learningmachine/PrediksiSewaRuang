import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import os

# Load model
MODEL_PATH = "model/model_rf_hargapenawaran.joblib"
if os.path.exists(MODEL_PATH):
    model_rf = joblib.load(MODEL_PATH)
    st.session_state["model_rf"] = model_rf
else:
    st.warning("⚠️ Model belum tersedia!")

# Sidebar: pilih modul
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Evaluasi Model", "Prediksi"])

# ==========================
# Halaman 1: Beranda
# ==========================
if page == "Beranda":
    st.title("🏠 Beranda")
    st.markdown("""
    Selamat datang di aplikasi Prediksi Harga Penawaran.  
    Gunakan modul **Evaluasi Model** untuk mengecek performa model.  
    Gunakan modul **Prediksi** untuk menghitung harga baru dari input variabel.
    """)

# ==========================
# Halaman 2: Evaluasi Model
# ==========================
elif page == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    uploaded_file = st.file_uploader("Upload dataset CSV/XLSX dengan kolom target 'HARGAPENAWARAN'", type=["csv","xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
        
        # Cek kolom target
        if "HARGAPENAWARAN" not in df.columns:
            st.error("Kolom target 'HARGAPENAWARAN' tidak ditemukan!")
        else:
            X = df.drop(columns=["HARGAPENAWARAN"])
            y = df["HARGAPENAWARAN"]
            X = X.loc[:, X.nunique() > 1]
            if "model_rf" in st.session_state:
                model_rf = st.session_state["model_rf"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)
                y_train_pred = model_rf.predict(X_train)
                y_test_pred = model_rf.predict(X_test)
                
                results = pd.DataFrame([{
                    'R2_in_sample': model_rf.score(X_train, y_train),
                    'R2_out_sample': model_rf.score(X_test, y_test),
                    'RMSE_in_sample': np.sqrt(np.mean((y_train - y_train_pred)**2)),
                    'RMSE_out_sample': np.sqrt(np.mean((y_test - y_test_pred)**2)),
                    'MAE_in_sample': np.mean(np.abs(y_train - y_train_pred)),
                    'MAE_out_sample': np.mean(np.abs(y_test - y_test_pred))
                }])
                st.dataframe(results.T)

# ==========================
# Halaman 3: Prediksi
# ==========================
elif page == "Prediksi":
    st.title("💡 Prediksi Harga")
    
    if "model_rf" not in st.session_state:
        st.warning("⚠️ Model belum tersedia!")
    else:
        model_rf = st.session_state["model_rf"]

        # Load feature columns
        FEATURE_PATH = "model/feature_columns.joblib"
        if os.path.exists(FEATURE_PATH):
            feature_cols = joblib.load(FEATURE_PATH)
        else:
            st.warning("⚠️ Feature columns belum tersedia! Pastikan file 'feature_columns.joblib' ada di folder model/")
            st.stop()

        # Input variabel otomatis berdasarkan feature columns
        st.subheader("Input Variabel")
        input_data = {}
        for col in feature_cols:
            val = st.number_input(f"{col}", value=0.0)
            input_data[col] = val
        input_df = pd.DataFrame([input_data])

        # Prediksi harga
        if st.button("Prediksi Harga"):
            try:
                pred_harga = model_rf.predict(input_df)[0]
                st.success(f"💰 Prediksi Harga: {pred_harga:,.2f}")

                # Top-5 similarity (gunakan dataset dummy atau dataset asli)
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np

                scaler = StandardScaler()
                # Dummy X untuk similarity; bisa diganti dengan dataset asli jika ada
                X_dummy = pd.DataFrame(np.random.rand(100, len(feature_cols)), columns=feature_cols)
                X_scaled = scaler.fit_transform(X_dummy)
                input_scaled = scaler.transform(input_df)
                sim_matrix = cosine_similarity(X_scaled, input_scaled)
                top5_idx = np.argsort(sim_matrix[:,0])[::-1][:5]
                st.subheader("Top-5 Similar Data Points (Index & Score)")
                top5_df = pd.DataFrame({
                    "Index": top5_idx,
                    "Similarity": sim_matrix[top5_idx,0]
                })
                st.dataframe(top5_df)
            except Exception as e:
                st.error(f"⚠️ Terjadi error saat prediksi: {e}")

