import pandas as pd
import streamlit as st
import altair as alt
import numpy as np 
import joblib
alt.data_transformers.disable_max_rows()
st.set_page_config(page_title = "Maintenance Predictions", page_icon= "⚙")
# Headings
st.title("Predictive Maintenance to detect a Machine Failure")
st.divider()

# Sidebar Input Details
st.sidebar.header('Input Details')
st.sidebar.divider()

def user_input_features():
    type_device = st.sidebar.selectbox('Type',('L','M','H'))
    air_temperature = st.sidebar.slider('Air Temperature (K)', 295.0, 305.0, 300.0)
    process_temperature = st.sidebar.slider('Process Temperature (K)', 305.0, 314.0, 310.0)
    rotational_speed = st.sidebar.slider("Rotational Speed (RPM)", 1168, 2886, 1500)
    torque = st.sidebar.slider("Torque (N-m)", 3.5, 77.0, 40.0)
    tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 253, 108)
    data = {'Type': type_device,
            'Air Temperature': air_temperature,
            'Process Temperature': process_temperature,
            'Rotational Speed': rotational_speed,
            'Torque': torque,
            'Tool wear': tool_wear
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
input_df_copy = input_df.copy()




input_df = input_df.drop(['Type', 'Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque'], axis = 1)

model = joblib.load("predictive_maintenance.pkl")
prediction = model.predict(input_df)
prediction_probability = model.predict_proba(input_df)


st.subheader('Prediction')
if prediction == 0:
    st.success('No Maintenance Required', icon="✅")
    st.write(f"Probability: **{list(prediction_probability)[0][0]:.2%}**")
else:
    st.error('Maintenance Needed', icon="🚨")
    st.write(f"Probability: **{list(prediction_probability)[0][1]:.2%}**")




st.write("---")

# Disclaimer
st.divider()
st.caption("""
        _This web app is intended for practical and showcase purposes only._
            
""")
