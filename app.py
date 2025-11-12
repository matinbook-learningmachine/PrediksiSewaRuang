# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Import model tambahan untuk evaluasi
from sklearn.linear_model import LinearRegression, ElasticNet, PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

# Import fungsi OSM
from my_osm_functions import (
    visualisasi_peta_adaptif_network,
    ambil_landuse_per_titik_fix,
    plot_landuse_titik_dan_polygon_enhanced
)

# Untuk peta interaktif
from streamlit_folium import st_folium

# ============================================================
# 🔹 Load model & feature columns
# ============================================================
MODEL_PATH = "model/model_rf_hargapenawaran.joblib"
FEATURE_PATH = "model/feature_columns.joblib"

if os.path.exists(MODEL_PATH):
    model_rf = joblib.load(MODEL_PATH)
    st.session_state["model_rf"] = model_rf

if os.path.exists(FEATURE_PATH):
    feature_cols = joblib.load(FEATURE_PATH)
    st.session_state["feature_cols"] = feature_cols

# ============================================================
# 🔹 Sidebar: Navigasi
# ============================================================
st.sidebar.title("📌 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["Beranda", "Evaluasi Model", "Prediksi", "Analisis OSM"]
)

# ============================================================
# Halaman 1: Beranda
# ============================================================
if page == "Beranda":
    st.title("🏠 Beranda")
    st.markdown("""
    Selamat datang di aplikasi Prediksi Harga Penawaran.  
    Gunakan modul **Evaluasi Model** untuk mengecek performa model.  
    Gunakan modul **Prediksi** untuk menghitung harga baru dari input variabel.  
    Gunakan modul **Analisis OSM** untuk analisis jarak ke jalan & landuse per titik.
    """)

# ============================================================
# Halaman 2: Evaluasi Model
# ============================================================
elif page == "Evaluasi Model":
    st.title("📊 Evaluasi Model")
    uploaded_file = st.file_uploader(
        "Upload dataset CSV/XLSX dengan kolom target 'HARGAPENAWARAN'", 
        type=["csv","xlsx"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        if "HARGAPENAWARAN" not in df.columns:
            st.error("Kolom target 'HARGAPENAWARAN' tidak ditemukan!")
        else:
            target = 'HARGAPENAWARAN'
            X = df.drop(columns=[target]).loc[:, df.nunique() > 1]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

            models = {
                'Linear Regression': LinearRegression(),
                'Support Vector Regression': SVR(),
                'K-Nearest Neighbor': KNeighborsRegressor(),
                'Elastic Net': ElasticNet(random_state=77),
                'Passive Aggressive Regressor': PassiveAggressiveRegressor(random_state=77),
                'Random Forest': RandomForestRegressor(random_state=77),
                'Gradient Boosting': GradientBoostingRegressor(random_state=77),
                'LightGBM': LGBMRegressor(random_state=77),
                'XGBoost': xgb.XGBRegressor(random_state=77, verbosity=0)
            }

            results = pd.DataFrame(columns=[
                'Model',
                'R2_in_sample','R2_out_sample',
                'MSE_in_sample','MSE_out_sample',
                'RMSE_in_sample','RMSE_out_sample',
                'MAE_in_sample','MAE_out_sample',
                'MAPE_in_sample','MAPE_out_sample'
            ])

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred  = model.predict(X_test)

                    mse_in  = mean_squared_error(y_train, y_train_pred)
                    rmse_in = np.sqrt(mse_in)
                    mae_in  = mean_absolute_error(y_train, y_train_pred)
                    mape_in = mean_absolute_percentage_error(y_train, y_train_pred)
                    r2_in   = r2_score(y_train, y_train_pred)

                    mse_out  = mean_squared_error(y_test, y_test_pred)
                    rmse_out = np.sqrt(mse_out)
                    mae_out  = mean_absolute_error(y_test, y_test_pred)
                    mape_out = mean_absolute_percentage_error(y_test, y_test_pred)
                    r2_out   = r2_score(y_test, y_test_pred)

                    results = pd.concat([results, pd.DataFrame([{
                        'Model': name,
                        'R2_in_sample': r2_in,
                        'R2_out_sample': r2_out,
                        'MSE_in_sample': mse_in,
                        'MSE_out_sample': mse_out,
                        'RMSE_in_sample': rmse_in,
                        'RMSE_out_sample': rmse_out,
                        'MAE_in_sample': mae_in,
                        'MAE_out_sample': mae_out,
                        'MAPE_in_sample': mape_in,
                        'MAPE_out_sample': mape_out
                    }])], ignore_index=True)
                except Exception as e:
                    st.warning(f"Model {name} gagal dijalankan: {e}")

            st.subheader("Hasil Evaluasi Model")
            st.dataframe(results.sort_values(by='RMSE_out_sample'))

# ============================================================
# Halaman 3: Prediksi
# ============================================================
elif page == "Prediksi":
    st.title("💡 Prediksi Harga")

    if "model_rf" not in st.session_state or "feature_cols" not in st.session_state:
        st.warning("⚠️ Model atau feature columns belum tersedia!")
    else:
        model_rf = st.session_state["model_rf"]
        feature_cols = st.session_state["feature_cols"]

        st.subheader("Input Variabel")
        input_data = {col: st.number_input(f"{col}", value=0.0) for col in feature_cols}
        input_df = pd.DataFrame([input_data])

        if st.button("Prediksi Harga"):
            try:
                pred_harga = model_rf.predict(input_df)[0]
                st.success(f"💰 Prediksi Harga: {pred_harga:,.2f}")

                scaler = StandardScaler()
                X_dummy = pd.DataFrame(np.random.rand(100, len(feature_cols)), columns=feature_cols)
                X_scaled = scaler.fit_transform(X_dummy)
                input_scaled = scaler.transform(input_df)
                sim_matrix = cosine_similarity(X_scaled, input_scaled)
                top5_idx = np.argsort(sim_matrix[:,0])[::-1][:5]

                st.subheader("Top-5 Similar Data Points (Index & Score)")
                st.dataframe(pd.DataFrame({"Index": top5_idx, "Similarity": sim_matrix[top5_idx,0]}))
            except Exception as e:
                st.error(f"⚠️ Terjadi error saat prediksi: {e}")

# ============================================================
# Halaman 4: Analisis OSM
# ============================================================
elif page == "Analisis OSM":
    st.title("🗺️ Analisis OSM (Jalan & Landuse)")

    uploaded_file_osm = st.file_uploader(
        "Upload dataset CSV/XLSX dengan kolom koordinat (Latitude & Longitude)",
        type=["csv","xlsx"]
    )

    if uploaded_file_osm:
        try:
            if uploaded_file_osm.name.endswith(".csv"):
                df_osm = pd.read_csv(uploaded_file_osm)
            else:
                df_osm = pd.read_excel(uploaded_file_osm)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        st.subheader("Pilih Kolom Koordinat")
        col_lat = st.selectbox("Kolom Latitude", options=df_osm.columns)
        col_lon = st.selectbox("Kolom Longitude", options=df_osm.columns)

        st.subheader("Menghitung jarak ke jalan & polygon wilayah administratif...")
        try:
            m_osm, df_jarak, df_detect = visualisasi_peta_adaptif_network(
                df_osm,
                summary_koordinat_v5_7=pd.DataFrame({
                    "Kolom": [col_lat, col_lon],
                    "In_Range": ["Latitude","Longitude"],
                    "Status": ["Koordinat valid","Koordinat valid"]
                }),
                cutoff_levels=[1000,2000,5000]
            )
            st.subheader("Peta Jalan & Titik")
            st_folium(m_osm, width=700, height=500)

            st.subheader("Data jarak ke jalan")
            st.dataframe(df_jarak)

            st.subheader("Mengambil Landuse per titik...")
            df_landuse, landuse_gdf = ambil_landuse_per_titik_fix(
                df_jarak.rename(columns={col_lat:"LATITUDE", col_lon:"LONGITUDE"})
            )
            st.dataframe(df_landuse)

            st.subheader("Peta Landuse")
            m_landuse = plot_landuse_titik_dan_polygon_enhanced(df_landuse, landuse_gdf)
            st_folium(m_landuse, width=700, height=500)

        except Exception as e:
            st.error(f"⚠️ Terjadi error saat analisis OSM: {e}")
