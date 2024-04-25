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
    
    st.subheader('Parameters description')
    st.write("""
        The dataset consists of the following features:
        * **Type**: Product quality variant with letters L, M, or H, and a variant-specific serial number.
        * **Air temperature [K]**: Generated using a random walk process, later normalized to a standard deviation of 2 K around 300 K.
        * **Process temperature [K]**: Generated using a random walk process, normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
        * **Rotational speed [rpm]**: Calculated from power of 2860 W, overlaid with normally distributed noise.
        * **Torque [Nm]**: Torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
        * **Tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
    """)
    
    st.subheader('The machine failure modes')
    st.write("""
        If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail 
        
        * **tool wear failure (TWF)**: the tool will be replaced of fail at a randomly selected tool wear time between 200 and 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
        * **heat dissipation failure (HDF)**: heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tool's rotational speed is below 1380 rpm. This is the case for 115 data points.
        * **power failure (PWF)**: the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
        * **overstrain failure (OSF)**: if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
        * **random failures (RNF)**: each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.
    """)
    
    st.subheader('About the Dataset')
    st.write("""
        The synthetic dataset provided in this application reflects real predictive maintenance encountered in the industry to the best of our knowledge. The dataset contains 10,000 data points with 14 features. It includes a mix of low, medium, and high-quality variants, each with a specific serial number. The features represent various parameters like air temperature, process temperature, rotational speed, torque, and tool wear.
    
        To explore the code and understand how our solution automates this process effectively, check out the project on [Kaggle](https://www.kaggle.com/code/atom1991/optimizing-operations-with-predictive-maintenance?kernelSessionId=146948811).
    """)



# Disclaimer
st.divider()
st.caption("""_This web app is intended for practical and showcase purposes only._""")
