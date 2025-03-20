# UTS x CUCAI Showcase Project - Generating Predictive Maintenance RUL Analytics for Nasa Turbofan Jet Engines Using LSTM, XGBoost, and Random Forest Architecture


## **RUL - Remaining Useful Lifetime | Predicted Using Sensor Data**


Arjan Waraich (waraicharjan97@gmail.com), Max Huddleston, Kushad Manikandan, Dora Li, Sidney Shu

## Python
Project Description:

See the Project Paper Attached in the Main Github Directory, Titled: FC_ProphetJet: Predictive Maintenance Modelling Using LSTM, Random Forest, and XGBoosting to Forecast RUL Metrics of NASA Turbofan Jet Engines

This project was built for the first annual CUCAI (Canadian Undergraduate Conference in Artificial Intelligence) in 2025, as a showcase project, by 6 UTS (University of Toronto Schools) High School students (one of two teams) who are passionate about AI, machine learning, and data analytics.

Using DL, we created a predictive model for failure classification of jet engines
for predictive maintenance in the aerospace industry, via LSTM (long short-term memory) models, Random Forest Regression, and XGBoost (extreme gradient boosting). Model output is a 1-dimensional feature which is predicted RUL (remaining useful lifetime) of the machine using time-series data provided - and how much of its operation at a specific power capacity remains, provided 21 different types of sensor analytics.
Adapted from NASAs CMAPPS Simulated Jet Engine Dataset.

## Dataset Details:

Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX
Fault Modes: TWO (HPC Degradation, Fan Degradation)

## Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, 'Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation', in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

## Sources Consulted:
 - Dataset: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data
 - Pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
 - Model Guidance: https://medium.com/@rohit.malhotra67/predictive-maintenance-on-nasas-turbofan-engine-degradation-dataset-cmapss-c066ee427931

"""
