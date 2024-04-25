# Predictive Maintenance to detect a Machine Failure

This project created to demonstrate how  predictive maintanance can be realized.
Prediction of a machine failure was done by machine learning algorithm. 

It utilizes a synthetic dataset with 10,000 data points and 14 features from Kaggle: [link](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) 
More information about the dataset is [here](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The application is built using a Random Forest model to classify whether the machine will experience failure or not based on the provided inputs.

## Stremlit app

The app can be found by the following link:
[Streamlit app](https://predictive-maintenance-kaggle-apptz2heajozb5kdkvelcxk.streamlit.app/)

## Parameters description

The dataset consists of the following features:
- `Type`: Product quality variant with letters L, M, or H, and a variant-specific serial number.
- `Air temperature [K]`: Generated using a random walk process, later normalized to a standard deviation of 2 K around 300 K.
- `Process temperature [K]`: Generated using a random walk process, normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
- `Rotational speed [rpm]`: Calculated from power of 2860 W, overlaid with normally distributed noise.
- `Torque [Nm]`: Torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values.
- `Tool wear [min]`: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.

## The machine failure modes

If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1. It is therefore not transparent to the machine learning method, which of the failure modes has caused the process to fail 

- `tool wear failure (TWF)`: the tool will be replaced of fail at a randomly selected tool wear time between 200 and 240 mins (120 times in our dataset). At this point in time, the tool is replaced 69 times, and fails 51 times (randomly assigned).
- `heat dissipation failure (HDF)`: heat dissipation causes a process failure, if the difference between air- and process temperature is below 8.6 K and the tool's rotational speed is below 1380 rpm. This is the case for 115 data points.
- `power failure (PWF)`: the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails, which is the case 95 times in our dataset.
- `overstrain failure (OSF)`: if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain. This is true for 98 datapoints.
- `random failures (RNF)`: each process has a chance of 0,1 % to fail regardless of its process parameters. This is the case for only 5 datapoints, less than could be expected for 10,000 datapoints in our dataset.


## About the Dataset

The synthetic dataset provided in this application reflects real predictive maintenance encountered in the industry to the best of our knowledge. The dataset contains 10,000 data points with 14 features. It includes a mix of low, medium, and high-quality variants, each with a specific serial number. The features represent various parameters like air temperature, process temperature, rotational speed, torque, and tool wear.

## Similar projects

- https://github.com/VivekAgrawl/predictive-maintenance-webapp
- https://github.com/RushikeshKothawade07/predictive-maintenance-ML/tree/main
