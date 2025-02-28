# DeepL1

"""
Arjan W.

[ ] "" > < \

## Python
Brief Project Description:
Using DL, creating a predictive model for multi-class failure of jet engines
for predictive maintenance in the aerospace industry, along with a regression for
RUL (remaining useful lifetime) of the machine using time-series data provided
from NASAs CMAPPS Simulated Jet Engine Dataset

Current Step: Train and Deploy Model (Feb 21st, 2025)

## Model Input: (Next Steps)
 - Append new data
 - Provide user-controlled in randomly generated data with deviations (for sake of demonstrating model accuracy) - (program random anomalies)
 - Inputs: # of model anomalies, # of rows to randomly generate, # deviation (provided a scale), time-series data
 - Potential UI

## Model Output:
Multi-Class Classification → Normal, minor fault, major fault, imminent failure.
Regression → Predicted remaining useful life (RUL) of the engine.


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
   (Pandas is quite nice for handling and preprocessing data for models to run on)
"""
