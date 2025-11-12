# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import folium_static
import folium
import os

# ================================
# 🔹 Load pre-trained model
# ================================
MODEL_PATH = "model/model_rf_hargapenawaran.joblib"

if os.path.exists(MODEL_PATH):
    model_rf = joblib.load(MODEL_PATH)
    st.session_state["model_rf"] = model_rf
else:
    st.warning("⚠️ File model_rf_hargapenawaran.joblib belum ada. Harap siapkan model pre-trained.")

# ================================
# 🔹 Sidebar: Upload Data
# ================================
st.sidebar.header("📤 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV dataset", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ Dataset berhasil diunggah!")
        st.write("Preview Dataset:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        st.stop()

    # ================================
    # 🔹 Tampilkan hasil model evaluation (results)
    # ================================
    st.header("📊 Evaluasi Model")
    st.info("Analisis dilakukan menggunakan Random Forest secara default.")

    # Normalisasi nama kolom: hapus spasi & uppercase
    df_columns_norm = {c.upper().replace(" ", ""): c for c in df.columns}
    target_norm = 'HARGAPENAWARAN'

    if target_norm not in df_columns_norm:
        st.error(f"❌ Kolom target '{target_norm}' tidak ditemukan di dataset!")
    else:
        target_col = df_columns_norm[target_norm]
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X = X.loc[:, X.nunique() > 1]

        if "model_rf" in st.session_state:
            model_rf = st.session_state["model_rf"]
            from sklearn.model_selection import train_test_split
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
            st.dataframe(results.T, width=600)

        # ================================
        # 🔹 Prediksi Data Baru
        # ================================
        st.header("💡 Prediksi Harga & Top-5 Similarity")
        st.subheader("Input Variabel")

        input_data = {}
        for col in X.columns:
            val = st.number_input(f"{col}", value=float(X[col].median()))
            input_data[col] = val
        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi Harga"):
            pred_harga = model_rf.predict(input_df)[0]
            st.success(f"💰 Prediksi Harga: {pred_harga:,.2f}")

            # Top-5 similarity
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            input_scaled = scaler.transform(input_df)
            sim_matrix = cosine_similarity(X_scaled, input_scaled)
            top5_idx = np.argsort(sim_matrix[:,0])[::-1][:5]
            st.subheader("Top-5 Similar Data Points (Index & Score)")
            top5_df = pd.DataFrame({
                "Index": top5_idx,
                "Similarity": sim_matrix[top5_idx,0]
            })
            st.dataframe(top5_df)

        # ================================
        # 🔹 Fitur Koordinat Baru
        # ================================
        st.header("📍 Analisis Koordinat Baru")
        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            st.warning("⚠️ Kolom 'LATITUDE' atau 'LONGITUDE' tidak ditemukan di dataset!")
        else:
            lat = st.number_input("Latitude", value=float(df['LATITUDE'].median()))
            lon = st.number_input("Longitude", value=float(df['LONGITUDE'].median()))

            if st.button("Hitung Jarak & Landuse"):
                try:
                    from my_osm_functions import visualisasi_peta_adaptif_network, summary_koordinat_v5_7

                    tmp_df = pd.DataFrame([{'LATITUDE': lat, 'LONGITUDE': lon}])
                    m, hasil_df, df_detect = visualisasi_peta_adaptif_network(
                        tmp_df, summary_koordinat_v5_7=summary_koordinat_v5_7
                    )
                    st.write("Hasil jarak ke jalan:")
                    st.dataframe(hasil_df[['LATITUDE','LONGITUDE','Jarak_ke_JalanUtama','Jarak_ke_JalanSekunder']])
                    st.write("Landuse / POI info:")
                    st.dataframe(df_detect)

                    st.subheader("🗺️ Peta Interaktif")
                    folium_static(m)
                except Exception as e:
                    st.error(f"❌ Gagal menghitung jarak/landuse: {e}")
