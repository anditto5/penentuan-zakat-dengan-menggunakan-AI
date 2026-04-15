import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="AI Zakat Dashboard",
    page_icon="🕌",
    layout="wide"
)

st.title("🕌 AI Zakat Eligibility Dashboard")
st.caption("Sistem prediksi mustahik berbasis Machine Learning")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("📥 Input Data Calon Mustahik")

income = st.sidebar.number_input("Pendapatan (IDR)", 0, 10000000, 2000000)
dependents = st.sidebar.slider("Jumlah Tanggungan", 0, 10, 2)
household_size = st.sidebar.slider("Anggota Keluarga", 1, 10, 4)
house_condition = st.sidebar.selectbox("Kondisi Rumah (1=buruk,5=baik)", [1,2,3,4,5])

water = st.sidebar.selectbox("Akses Air Bersih", ["no","yes"])
electricity = st.sidebar.selectbox("Akses Listrik", ["no","yes"])
previous_aid = st.sidebar.selectbox("Pernah Dapat Bantuan", ["no","yes"])

employment = st.sidebar.selectbox("Status Pekerjaan", ["unemployed","employed"])
education = st.sidebar.selectbox("Pendidikan", ["SD","SMP","SMA","Diploma","Sarjana"])
region = st.sidebar.selectbox("Wilayah", ["Desa","Kota"])

# =========================
# PREPROCESS
# =========================
def preprocess():
    df = pd.DataFrame(columns=model.feature_names_in_)
    df.loc[0] = 0  # default semua 0

    df.loc[0, "income"] = income
    df.loc[0, "household_size"] = household_size
    df.loc[0, "dependents"] = dependents
    df.loc[0, "house_condition"] = house_condition

    df.loc[0, "access_to_water"] = 1 if water == "yes" else 0
    df.loc[0, "access_to_electricity"] = 1 if electricity == "yes" else 0
    df.loc[0, "previous_aid"] = 1 if previous_aid == "yes" else 0

    df.loc[0, "employment_unemployed"] = 1 if employment == "unemployed" else 0

    df.loc[0, "education_SMP"] = 1 if education == "SMP" else 0
    df.loc[0, "education_SMA"] = 1 if education == "SMA" else 0
    df.loc[0, "education_Diploma"] = 1 if education == "Diploma" else 0
    df.loc[0, "education_Sarjana"] = 1 if education == "Sarjana" else 0

    df.loc[0, "region_rural"] = 1 if region == "Desa" else 0
    df.loc[0, "region_suburban"] = 0  # kalau tidak dipakai

    return df

input_df = preprocess()

# =========================
# PREDICTION
# =========================
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# =========================
# LAYOUT DASHBOARD
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("💰 Income", f"Rp {income:,}")
col2.metric("👨‍👩‍👧 Tanggungan", dependents)
col3.metric("📊 Probabilitas", f"{probability:.2f}")

st.divider()

# =========================
# RESULT SECTION
# =========================
left, right = st.columns([2,1])

with left:
    st.subheader("📊 Hasil Analisis AI")

    if prediction == 1:
        st.error("❌ TERIDENTIFIKASI: MUSTAHIK (Layak menerima zakat)")
    else:
        st.success("✅ NON-MUSTAHIK")

    # Probability gauge (bar chart style)
    fig = px.bar(
        x=["Probability"],
        y=[probability],
        range_y=[0,1],
        text=[round(probability,2)],
        title="Tingkat Kelayakan Zakat"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🎯 Prioritas Bantuan")

    if probability > 0.4:
        st.warning("🔥 PRIORITAS TINGGI")
        priority = "High"
    elif probability > 0.2:
        st.info("⚠️ PRIORITAS SEDANG")
        priority = "Medium"
    else:
        st.success("🟢 PRIORITAS RENDAH")
        priority = "Low"

    st.write(f"Kategori: **{priority}**")

# =========================
# INSIGHT SECTION
# =========================
st.divider()
st.subheader("🧠 Insight AI")

insights = []

if income < 2000000:
    insights.append("Pendapatan rendah meningkatkan kemungkinan mustahik")

if dependents > 1:
    insights.append("Jumlah tanggungan tinggi → beban ekonomi besar")

if water == "no":
    insights.append("Tidak ada akses air bersih → indikator kemiskinan")

if employment == "unemployed":
    insights.append("Tidak bekerja → risiko ekonomi tinggi")

for i in insights:
    st.write("🔹", i)

# =========================
# FOOTER
# =========================
st.caption("AI Zakat System • Prototype for Social Impact AI")