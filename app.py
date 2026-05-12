
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- 1. Cargar el modelo y el escalador (si es necesario) ---
# Cargamos el modelo (RandomForestClassifier) que guardamos
model = joblib.load('modelo_desercion.pkl')

# Es crucial usar el mismo StandardScaler con el que se entrenó el modelo
# Si no lo guardamos, necesitamos crear uno nuevo y 'ajustarlo' con algunos datos
# Para este ejemplo, supondremos que X_train (o un subconjunto) está disponible
# En un escenario real, deberíamos guardar el 'scaler' junto con el modelo.

# Para fines de demostración, recrearemos un scaler similar. 
# IMPORTANTE: En un entorno de producción, DEBERÍAS guardar y cargar el scaler original.
# Aquí usamos X_train que ya tenemos en memoria (asumiendo que se ejecutó previamente)
# o podrías cargarlo desde un archivo si lo guardaste.

# Recrear el scaler usando los datos de entrenamiento originales (no escalados) para ajustarlo
# Asumimos que X_train existe del notebook anterior
scaler = StandardScaler()
scaler.fit(X_train) # Ajustar con los datos originales no escalados

# --- 2. Configuración de la aplicación Streamlit ---
st.set_page_config(page_title='Predicción de Deserción Estudiantil', layout='centered')
st.title('📊 Predicción de Deserción Estudiantil')
st.write('Ingrese los datos del estudiante para predecir la probabilidad de deserción.')

# --- 3. Inputs para las variables ---
st.header('Datos del Estudiante')

# Sliders para variables numéricas
edad = st.slider('Edad', 18, 30, 22)
promedio = st.slider('Promedio (0-10)', 5.0, 9.5, 7.5)
asistencia = st.slider('Asistencia (%)', 60, 100, 85)
horas_estudio = st.slider('Horas de Estudio Semanales', 5, 40, 20)
uso_plataforma = st.slider('Uso de Plataforma (%)', 30, 100, 70)
materias_perdidas = st.slider('Materias Perdidas', 0, 5, 1)
nivel_socioeconomico = st.slider('Nivel Socioeconómico (1=Bajo, 5=Alto)', 1, 5, 3)

# Selectbox para variables categóricas
trabaja_map = {'No': 0, 'Sí': 1}
trabaja_display = st.selectbox('¿El estudiante trabaja?', list(trabaja_map.keys()))
trabaja = trabaja_map[trabaja_display]

acceso_internet_map = {'No': 0, 'Sí': 1}
acceso_internet_display = st.selectbox('¿Tiene acceso a internet?', list(acceso_internet_map.keys()))
acceso_internet = acceso_internet_map[acceso_internet_display]

# --- 4. Botón de Predicción ---
if st.button('Realizar Predicción'):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame([{
        'edad': edad,
        'promedio': promedio,
        'asistencia': asistencia,
        'horas_estudio': horas_estudio,
        'uso_plataforma': uso_plataforma,
        'materias_perdidas': materias_perdidas,
        'nivel_socioeconomico': nivel_socioeconomico,
        'trabaja': trabaja,
        'acceso_internet': acceso_internet
    }])

    # Escalar los datos de entrada usando el mismo scaler que se usó para el entrenamiento
    input_data_scaled = scaler.transform(input_data)

    # Realizar la predicción
    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[0][1] # Probabilidad de deserción (clase 1)

    st.header('Resultado de la Predicción')
    if prediction == 1:
        st.error(f'🚨 ¡ALERTA! El estudiante tiene una ALTA probabilidad de desertar ({prediction_proba:.2%})')
        st.write('**Estrategias recomendadas:** Ofrecer tutorías personalizadas, asesoramiento psicológico, apoyo financiero, y seguimiento académico continuo.')
    else:
        st.success(f'✅ El estudiante tiene una BAJA probabilidad de desertar ({prediction_proba:.2%})')
        st.write('**Observación:** Continúe monitoreando el progreso académico y el bienestar general del estudiante.')

    st.markdown('---')
    st.info('Esta predicción es un indicador y no un diagnóstico definitivo. Siempre se recomienda la intervención humana y el análisis contextual.')

