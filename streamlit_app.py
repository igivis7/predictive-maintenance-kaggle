import pandas as pd
import streamlit as st
import altair as alt
import numpy as np 
import joblib
alt.data_transformers.disable_max_rows()

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

tab1, tab2 = st.tabs(['Machine Failure Prediction', 'Data info'])
# Tab 1
with tab1:
    st.info('Adjust the sliders or select values in the **sidebar** to input essential operational data', icon="ℹ️")
    st.subheader('Input Details')
    st.write(f"""
            * **Type:** {input_df['Type'].values[0]}
            * **Air Temperature:** {input_df['Air Temperature'].values[0]} K
            * **Process Temperature:** {input_df['Process Temperature'].values[0]} K
            * **Rotational Speed:** {input_df['Rotational Speed'].values[0]} RPM
            * **Torque:** {input_df['Torque'].values[0]} Nm
            * **Tool Wear:** {input_df['Tool wear'].values[0]} min
            """)

# Feature Engineering
input_df['Power'] = 2 * np.pi * input_df['Rotational Speed'] * input_df['Torque'] / 60
input_df['temp_diff'] = input_df['Process Temperature'] - input_df['Air Temperature']
input_df['Type_H'] = 0
input_df['Type_L'] = 0
input_df['Type_M'] = 0
if input_df['Type'].values == 'L':
    input_df['Type_L'] = 1
elif input_df['Type'].values == 'M':
    input_df['Type_M'] = 1
else:
    input_df['Type_H'] = 1

input_df = input_df.drop(['Type', 'Air Temperature', 'Process Temperature', 'Rotational Speed', 'Torque'], axis = 1)

model = joblib.load("predictive_maintenance.pkl")
prediction = model.predict(input_df)
prediction_probability = model.predict_proba(input_df)

# Tab 1
with tab1:
    st.subheader('Prediction')
    if prediction == 0:
        st.success('No Maintenance Required', icon="✅")
        st.write(f"Probability: **{list(prediction_probability)[0][0]:.2%}**")
    else:
        st.error('Maintenance Needed', icon="🚨")
        st.write(f"Probability: **{list(prediction_probability)[0][1]:.2%}**")


# Tab 2
with tab2:
    st.write("""
        Welcome to an innovative project designed to enhance maintenance efficiency for manufacturing companies! Our machine learning model, incorporated into a user-friendly web app, predicts maintenance needs in real time by analyzing data from industrial devices. This proactive solution empowers companies to tackle issues before they cause downtime and increased costs.

        The key components of the project include:

        * **Type:** Products are categorized as low (L), medium (M), or high (H) quality variants, each with a specific serial number.
        * **Air Temperature [K]:** The temperature of the surrounding air, measured in Kelvin.
        * **Process Temperature [K]:** The temperature of the manufacturing process, measured in Kelvin.
        * **Rotational Speed [RPM]:** The speed at which the device rotates, measured in revolutions per minute.
        * **Torque [Nm]:** The applied torque to the device, measured in Newton-meters.
        * **Tool Wear [min]:** The duration of tool usage, measured in minutes.

        To explore the code and understand how our solution automates this process effectively, check out the project on [Kaggle](https://www.kaggle.com/code/atom1991/optimizing-operations-with-predictive-maintenance?kernelSessionId=146948811).
    """)



# Disclaimer
st.divider()
st.caption("""_This web app is intended for practical and showcase purposes only._""")
