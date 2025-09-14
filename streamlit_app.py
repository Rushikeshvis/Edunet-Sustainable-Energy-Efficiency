import streamlit as st
import pandas as pd
import joblib
import datetime
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("Model assets not found. Please run the 'train_and_save_model.py' script first.")
        return None, None, None

model, scaler, model_columns = load_assets()

st.set_page_config(page_title="Solar Forecaster", page_icon="☀️")
st.title('☀️ Solar Power Forecaster')
st.sidebar.header('Input Parameters')

if model is not None:
    date = st.sidebar.date_input("Date", datetime.date.today())
    hour = st.sidebar.slider("Hour of the day", 0, 23, 12)
    inverter_id_suffix = model_columns[-1].split('_')[-1] # Inferring an example inverter
    
    ambient_temp = st.sidebar.number_input("Ambient Temp (°C)", value=25.0)
    module_temp = st.sidebar.number_input("Module Temp (°C)", value=40.0)
    irradiation = st.sidebar.number_input("Irradiation (kW/m²)", value=0.5)

    if st.sidebar.button('Predict', type="primary"):
        input_data = {
            'MONTH': date.month,
            'DAY_OF_WEEK': date.weekday(),
            'HOUR': hour,
            'AMBIENT_TEMPERATURE': ambient_temp,
            'MODULE_TEMPERATURE': module_temp,
            'IRRADIATION': irradiation,
            f'INVERTER_ID_{inverter_id_suffix}': 1 
        }
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        numerical_features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'MONTH', 'DAY_OF_WEEK', 'HOUR']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        prediction = model.predict(input_df)

        st.subheader('⚡ Predicted Power Output')
        st.metric(label="AC Power (W)", value=f"{prediction[0]:.2f}")