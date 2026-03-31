
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder for use in app

model = joblib.load("modelo_gravedad.pkl")
robust_scaler = joblib.load("robust_scaler_gravedad.pkl")
standard_scaler = joblib.load("standard_scaler_gravedad.pkl")
encoders_classes = joblib.load("encoders_classes.pkl") # Load the classes

# Reconstruct LabelEncoders using the loaded classes
encoders = {}
for col, classes in encoders_classes.items():
    le = LabelEncoder()
    le.fit(classes) # Fit the encoder with the loaded classes
    encoders[col] = le

# Use the loaded classes for selectbox options
opciones = {col: classes for col, classes in encoders_classes.items()}

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
        # These are original numbers, no encoding needed for them directly
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
    # Prepare the input dictionary, ensuring column order matches training data 'X'
    # (excluding 'FECHA ACCIDENTE', 'HORA ACCIDENTE' which are dropped from X)
    entrada = {
        # Categorical features, label encoded
        "CLASE ACCIDENTE": encoders["CLASE ACCIDENTE"].transform([clase])[0],
        "DESCRIPCIÓN COLISIONANTE": encoders["DESCRISubtask: Dynamically populate selectbox options in Streamlit app from loaded encoder classes.CIÓN COLISIONANTE"].transform([colisionante])[0],
        "DESCRIPCIÓN OBJETO FIJO": encoders["DESCRIPCIÓN OBJETO FIJO"].transform([objeto_fijo])[0],
        "ZONA ACCIDENTE": encoders["ZONA ACCIDENTE"].transform([zona])[0],
        "DESCRIPCIÓN DE ESTADO": encoders["DESCRIPCIÓN DE ESTADO"].transform([estado_proceso])[0],
        "ÁREA ACCIDENTE": encoders["ÁREA ACCIDENTE"].transform([area])[0],
        "SECTOR ACCIDENTE": encoders["SECTOR ACCIDENTE"].transform([sector])[0],
        "DESCRIPCIÓN LOCALIZACIÓN": encoders["DESCRIPCIÓN LOCALIZACIÓN"].transform([localizacion])[0],
        "ESTADO CLIMA": encoders["ESTADO CLIMA"].transform([clima])[0],
        "MUNICIPIO": encoders["MUNICIPIO"].transform([municipio])[0],

        # Numerical features (victim counts, temporal features)
        "NUMERO VICTIMA PEATÓN": float(peatones),
        "NUMERO VICTIMA ACOMPAÑANTE": float(acompanantes),
        "NUMERO VICTIMA PASAJERO": float(pasajeros),
        "NUMERO VICTIMA CONDUCTOR": float(conductores),
        "NUMERO VICTIMA HERIDO": float(heridos),
        "NUMERO VICTIMA MUERTO": float(muertos),
        "AÑO": float(anio),
        "MES": float(mes),
        "DIA_SEMANA": float(dia_semana),
        "HORA": float(hora),
        "FRANJA_HORARIA": float(encoders["FRANJA_HORARIA"].transform([franja])[0]),
    }

    # Ensure the order of columns in df_entrada matches X_train used for scaling and training
    # The order of columns in X was determined by df_clean.drop(columns=...)
    # and X_train was a subset of this X. It is critical that df_entrada columns are in the same order.
    # We can get the order from X.columns directly from the notebook kernel context (X.columns.tolist())
    # X.columns.tolist(): ['CLASE ACCIDENTE', 'DESCRIPCIÓN COLISIONANTE', 'DESCRIPCIÓN OBJETO FIJO', 'ZONA ACCIDENTE', 'DESCRIPCIÓN DE ESTADO', 'ÁREA ACCIDENTE', 'SECTOR ACCIDENTE', 'DESCRIPCIÓN LOCALIZACIÓN', 'ESTADO CLIMA', 'MUNICIPIO', 'NUMERO VICTIMA PEATÓN', 'NUMERO VICTIMA ACOMPAÑANTE', 'NUMERO VICTIMA PASAJERO', 'NUMERO VICTIMA CONDUCTOR', 'NUMERO VICTIMA HERIDO', 'NUMERO VICTIMA MUERTO', 'AÑO', 'MES', 'DIA_SEMANA', 'HORA', 'FRANJA_HORARIA']
    feature_order = [
        'CLASE ACCIDENTE', 'DESCRIPCIÓN COLISIONANTE', 'DESCRIPCIÓN OBJETO FIJO',
        'ZONA ACCIDENTE', 'DESCRIPCIÓN DE ESTADO', 'ÁREA ACCIDENTE',
        'SECTOR ACCIDENTE', 'DESCRIPCIÓN LOCALIZACIÓN', 'ESTADO CLIMA', 'MUNICIPIO',
        'NUMERO VICTIMA PEATÓN', 'NUMERO VICTIMA ACOMPAÑANTE', 'NUMERO VICTIMA PASAJERO',
        'NUMERO VICTIMA CONDUCTOR', 'NUMERO VICTIMA HERIDO', 'NUMERO VICTIMA MUERTO',
        'AÑO', 'MES', 'DIA_SEMANA', 'HORA', 'FRANJA_HORARIA'
    ]

    df_entrada = pd.DataFrame([entrada], columns=feature_order)

    # Add a check for victim counts and provide a hint if they are zero
    if entrada["NUMERO VICTIMA HERIDO"] == 0 and entrada["NUMERO VICTIMA MUERTO"] == 0:
        st.info("💡 **Nota:** Con 0 víctimas heridas y 0 fallecidas, el modelo tiende a predecir 'solo daños materiales'. Intente con valores distintos de cero en 'Víctimas Herido' o 'Víctimas Muerto' para ver otras predicciones.")

    # Apply RobustScaler first, then StandardScaler
    df_robust_scaled = robust_scaler.transform(df_entrada)
    df_standard_scaled = standard_scaler.transform(df_robust_scaled)

    prediccion = model.predict(df_standard_scaled)[0]
    probabilidad = model.predict_proba(df_standard_scaled)[0]

    st.markdown("---")
    if prediccion == 1:
        st.error("⚠️ El accidente presenta **víctimas** (heridos o muertos).")
    else:
        st.success("✅ El accidente involucra **solo daños materiales**.")

    st.markdown(f"**Probabilidad de accidente con víctimas:** {probabilidad[1]*100:.1f}%")
    st.markdown(f"**Probabilidad de solo daños materiales:** {probabilidad[0]*100:.1f}%")
