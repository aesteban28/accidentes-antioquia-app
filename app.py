
import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("modelo_gravedad.pkl")
scaler = joblib.load("scaler_gravedad.pkl")
encoders = joblib.load("encoders.pkl")

opciones = {
    "CLASE ACCIDENTE": ["ATROPELLO", "CAÍDA OCUPANTE", "CHOQUE", "INCENDIO", "NO REPORTADA", "OTRO", "VOLCAMIENTO"],
    "DESCRIPCIÓN COLISIONANTE": ["NO REPORTADO", "OBJETO FIJO", "SEMOVIENTE", "Semoviente", "VEHÍCULO"],
    "DESCRIPCIÓN OBJETO FIJO": ["ÁRBOL", "BARANDA", "INMUEBLE", "MURO", "NO REPORTADO", "OTRO", "POSTE", "VALLA, SEÑAL", "VEHÍCULO ESTACIONADO"],
    "ZONA ACCIDENTE": ["DEPORTIVA", "ESCOLAR", "Escolar", "HOSPITALARIA", "MILITAR", "NO REPORTADO", "PRIVADA", "TURÍSTICA"],
    "DESCRIPCIÓN DE ESTADO": ["CONCILIADO", "FALLADO", "PEND. DECLARACIÓN", "PEND. FALLO", "PEND. NOTIFICACIÓN", "Pend. Declaración", "Pend. Notificación"],
    "ÁREA ACCIDENTE": ["NO REPORTADO", "Rural", "RURAL", "Urbana", "URBANA"],
    "SECTOR ACCIDENTE": ["COMERCIAL", "Comercial", "INDUSTRIAL", "Industrial", "NO REPORTADO", "No Reportado", "RESIDENCIAL", "Residencial"],
    "DESCRIPCIÓN LOCALIZACIÓN": ["GLORIETA", "INTERSECCIÓN", "Interseccion", "LOTE O PREDIO", "NO REPORTADO", "PASO A NIVEL", "PASO ELEVADO", "PASO INFERIOR", "PUENTE", "TRAMO DE VÍA", "Tramo de via", "VÍA PEATONAL", "VÍA TRONCAL"],
    "ESTADO CLIMA": ["LLUVIA", "Lluvia", "NIEBLA", "NORMAL", "Normal"],
    "MUNICIPIO": ["ABEJORRAL", "ALEJANDRÍA", "AMAGA", "ANTIOQUIA", "ANZA", "BETANIA", "BURITICA", "CAICEDO", "CARAMANTA", "COCORNA", "CONCORDIA", "CÁCERES", "FREDONIA", "GRANADA", "GUARNE", "GUATAPE", "ITUANGO", "JARDÍN", "JERICÓ", "LA UNIÓN", "LIBORINA", "NO REPORTA", "PEÑOL", "PUEBLO RICO", "SAN JERÓNIMO", "SAN LUIS", "SAN VICENTE", "SANTA BARBARA - ANT", "SANTABÁRBARA", "SOPETRAN", "STAFE DE ANTIOQUIA", "TÁMESIS", "TITIRIBÍ", "VALPARAÍSO", "VENECIA"],
    "FRANJA_HORARIA": ["mañana", "noche", "tarde"]
}

st.set_page_config(page_title="Predicción de Gravedad de Accidentes", layout="centered")
st.title("Predicción de Gravedad de Accidentes de Tránsito")
st.markdown("**Departamento de Antioquia, Colombia**")
st.markdown("---")
st.markdown("Complete los campos para predecir si un accidente involucra víctimas.")

with st.form("formulario"):
    col1, col2 = st.columns(2)

    with col1:
        clase = st.selectbox("Clase de Accidente", opciones["CLASE ACCIDENTE"])
        colisionante = st.selectbox("Descripción Colisionante", opciones["DESCRIPCIÓN COLISIONANTE"])
        objeto_fijo = st.selectbox("Descripción Objeto Fijo", opciones["DESCRIPCIÓN OBJETO FIJO"])
        zona = st.selectbox("Zona del Accidente", opciones["ZONA ACCIDENTE"])
        estado_proceso = st.selectbox("Descripción de Estado", opciones["DESCRIPCIÓN DE ESTADO"])
        area = st.selectbox("Área del Accidente", opciones["ÁREA ACCIDENTE"])
        sector = st.selectbox("Sector del Accidente", opciones["SECTOR ACCIDENTE"])
        localizacion = st.selectbox("Descripción Localización", opciones["DESCRIPCIÓN LOCALIZACIÓN"])
        clima = st.selectbox("Estado del Clima", opciones["ESTADO CLIMA"])
        municipio = st.selectbox("Municipio", opciones["MUNICIPIO"])
        franja = st.selectbox("Franja Horaria", opciones["FRANJA_HORARIA"])

    with col2:
        peatones = st.number_input("Víctimas Peatón", min_value=0, max_value=10, value=0)
        acompanantes = st.number_input("Víctimas Acompañante", min_value=0, max_value=10, value=0)
        pasajeros = st.number_input("Víctimas Pasajero", min_value=0, max_value=25, value=0)
        conductores = st.number_input("Víctimas Conductor", min_value=0, max_value=5, value=0)
        heridos = st.number_input("Número de Heridos", min_value=0, max_value=25, value=0)
        muertos = st.number_input("Número de Muertos", min_value=0, max_value=5, value=0)
        anio = st.number_input("Año", min_value=2014, max_value=2030, value=2024)
        mes = st.number_input("Mes", min_value=1, max_value=12, value=1)
        dia_semana = st.number_input("Día de la Semana (0=Lunes, 6=Domingo)", min_value=0, max_value=6, value=0)
        hora = st.number_input("Hora del Accidente (0-23)", min_value=0, max_value=23, value=12)

    submitted = st.form_submit_button("Predecir Gravedad")

if submitted:
    entrada = {
        "CLASE ACCIDENTE": encoders["CLASE ACCIDENTE"].transform([clase])[0],
        "DESCRIPCIÓN COLISIONANTE": encoders["DESCRIPCIÓN COLISIONANTE"].transform([colisionante])[0],
        "DESCRIPCIÓN OBJETO FIJO": encoders["DESCRIPCIÓN OBJETO FIJO"].transform([objeto_fijo])[0],
        "ZONA ACCIDENTE": encoders["ZONA ACCIDENTE"].transform([zona])[0],
        "DESCRIPCIÓN DE ESTADO": encoders["DESCRIPCIÓN DE ESTADO"].transform([estado_proceso])[0],
        "ÁREA ACCIDENTE": encoders["ÁREA ACCIDENTE"].transform([area])[0],
        "SECTOR ACCIDENTE": encoders["SECTOR ACCIDENTE"].transform([sector])[0],
        "DESCRIPCIÓN LOCALIZACIÓN": encoders["DESCRIPCIÓN LOCALIZACIÓN"].transform([localizacion])[0],
        "ESTADO CLIMA": encoders["ESTADO CLIMA"].transform([clima])[0],
        "MUNICIPIO": encoders["MUNICIPIO"].transform([municipio])[0],
        "NUMERO VICTIMA PEATÓN": peatones,
        "NUMERO VICTIMA ACOMPAÑANTE": acompanantes,
        "NUMERO VICTIMA PASAJERO": pasajeros,
        "NUMERO VICTIMA CONDUCTOR": conductores,
        "NUMERO VICTIMA HERIDO": heridos,
        "NUMERO VICTIMA MUERTO": muertos,
        "AÑO": anio,
        "MES": mes,
        "DIA_SEMANA": dia_semana,
        "HORA": hora,
        "FRANJA_HORARIA": encoders["FRANJA_HORARIA"].transform([franja])[0],
    }

    df_entrada = pd.DataFrame([entrada])
    df_scaled = scaler.transform(df_entrada)
    prediccion = model.predict(df_scaled)[0]
    probabilidad = model.predict_proba(df_scaled)[0]

    st.markdown("---")
    if prediccion == 1:
        st.error("⚠️ El accidente presenta **víctimas** (heridos o muertos).")
    else:
        st.success("✅ El accidente involucra **solo daños materiales**.")

    st.markdown(f"**Probabilidad de accidente con víctimas:** {probabilidad[1]*100:.1f}%")
    st.markdown(f"**Probabilidad de solo daños materiales:** {probabilidad[0]*100:.1f}%")
