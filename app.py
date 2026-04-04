
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

model = joblib.load("modelo_rf_ambulancias.pkl")
columnas = joblib.load("columnas_modelo.pkl")

# Extraer lista de municipios desde las columnas del modelo
municipios = [col.replace("MUNICIPIO_", "") for col in columnas if col.startswith("MUNICIPIO_")]

st.set_page_config(page_title="Asignación de Ambulancias", layout="centered")
st.title("🚑 Sistema de Asignación de Ambulancias")
st.markdown("**Departamento de Antioquia, Colombia**")
st.markdown("---")

with st.form("formulario"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📍 Ubicación del Accidente")
        municipio = st.selectbox("Municipio", municipios)
        
        st.markdown("### 📅 Fecha y Hora")
        fecha = st.date_input("Fecha del accidente", datetime.now())
        hora = st.time_input("Hora del accidente", datetime.now().time())
    
    with col2:
        st.markdown("### 🚑 Ambulancias Disponibles")
        st.markdown("Ingrese cantidad disponible por municipio:")
        
        ambulancias = {}
        for mun in municipios[:10]:  # Mostrar primeros 10 como ejemplo
            ambulancias[mun] = st.number_input(f"{mun}", min_value=0, max_value=20, value=1, key=mun)
    
    submitted = st.form_submit_button("Asignar Ambulancia")

if submitted:
    # Extraer características temporales
    dia = fecha.day
    mes = fecha.month
    año = fecha.year
    dia_semana = fecha.weekday()
    hora_num = hora.hour
    
    # Determinar franja horaria
    franja_mañana = 1 if 6 <= hora_num < 12 else 0
    franja_tarde = 1 if 12 <= hora_num < 18 else 0
    franja_noche = 1 if hora_num >= 18 or hora_num < 6 else 0
    
    # Crear vector de entrada base
    entrada = {
        "dia": dia,
        "mes": mes,
        "año": año,
        "dia_semana": dia_semana,
        "hora_num": hora_num,
        "franja_mañana": franja_mañana,
        "franja_tarde": franja_tarde,
        "franja_noche": franja_noche
    }
    
    # Inicializar todos los municipios en 0 (one-hot encoding)
    for col in columnas:
        if col.startswith("MUNICIPIO_"):
            entrada[col] = 0
    
    # Activar el municipio seleccionado
    entrada[f"MUNICIPIO_{municipio}"] = 1
    
    # Crear DataFrame con el orden correcto de columnas
    df_entrada = pd.DataFrame([entrada], columns=columnas)
    
    # Predecir tiempo IDA
    tiempo_ida = model.predict(df_entrada)[0]
    
    st.markdown("---")
    st.success(f"⏱️ **Tiempo IDA estimado al municipio {municipio}:** {tiempo_ida:.2f} minutos")
    
    # Lógica de asignación
    st.markdown("### 🚑 Asignación Recomendada")
    
    if ambulancias.get(municipio, 0) > 0:
        st.info(f"✅ **Asignar ambulancia desde {municipio}** (mismo municipio, {ambulancias[municipio]} disponible(s))")
    else:
        st.warning(f"⚠️ No hay ambulancias disponibles en {municipio}. Calculando alternativas...")
        
        # Calcular tiempos desde otros municipios con ambulancias
        tiempos_alternativos = []
        for mun_origen, cant in ambulancias.items():
            if cant > 0 and mun_origen != municipio:
                entrada_alt = entrada.copy()
                # Resetear municipios
                for col in columnas:
                    if col.startswith("MUNICIPIO_"):
                        entrada_alt[col] = 0
                entrada_alt[f"MUNICIPIO_{mun_origen}"] = 1
                
                df_alt = pd.DataFrame([entrada_alt], columns=columnas)
                tiempo_alt = model.predict(df_alt)[0]
                tiempos_alternativos.append((mun_origen, tiempo_alt, cant))
        
        if tiempos_alternativos:
            tiempos_alternativos.sort(key=lambda x: x[1])
            mejor = tiempos_alternativos[0]
            st.success(f"✅ **Asignar ambulancia desde {mejor[0]}** (tiempo estimado: {mejor[1]:.2f} min, {mejor[2]} disponible(s))")
            
            st.markdown("**Otras opciones:**")
            for i, (mun, tiempo, cant) in enumerate(tiempos_alternativos[1:4], 2):
                st.text(f"{i}. {mun}: {tiempo:.2f} min ({cant} disponible(s))")
        else:
            st.error("❌ No hay ambulancias disponibles en ningún municipio.")
