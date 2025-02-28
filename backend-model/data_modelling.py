##

import pandas as pd
 

# [ ] "" > < \ (my keyboard is buggy, I need these code operators stored here temp.)

# Load training set for FD1
personal_fp_train_set = r"C:\Users\arjan\OneDrive_Arj\Desktop\Programming_Arjan\ICS3U Personal Project\CMAPSS Dataset\train_FD001.txt"

# Load the RUL (remaining useful lifetime) vector in respect for FD1

personal_fp_RUL = r"C:\Users\arjan\OneDrive_Arj\Desktop\Programming_Arjan\ICS3U Personal Project\CMAPSS Dataset\RUL_FD001.txt"

## These are the case sensitive column names that are provided from NASAs CMAPPS Dataset, specifically in the file: train_FD001.txt

jet_engine_data = ["unit", "cycle", "operational_setting_1", "operational_setting_2", "operational_setting_3",
                "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6", "sensor_7", 
                "sensor_8", "sensor_9", "sensor_10", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
                "sensor_15", "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21"]

dframe_train = pd.read_csv(personal_fp_train_set, sep=" ", header=None, names=jet_engine_data, engine="python") # get pandas to read and process the data - setting up a dataframe for the training data of FD001

dframe_train = dframe_train.dropna(axis=1, how="all") # I used .dropna(), which removes any rows that contain null values, because that can skew weightage and bias in the model

## assign each vector index under the categorized name of RUL

dframe_rul = pd.read_csv(personal_fp_RUL, header=None, names=['RUL']) # get pandas to read and process the data - setting up a dataframe for the RUL data of FD001

# adding a unit number to each row of the RUL dataframe - as its a vector, but also adjusting the index as its from 1 to N engine units

dframe_rul['unit'] = dframe_rul.index + 1

###### Targeting prediction of HPC fault occurences - high pressure compressors within jet enginges, which are crucial to the engine's operation

# assigning an INTEGER value to the unit number in both, train and RUL, datasets - avoid merge error later
dframe_train['unit'] = dframe_train['unit'].astype(int)
dframe_train['cycle'] = dframe_train['cycle'].astype(int)
dframe_rul['unit'] = dframe_rul['unit'].astype(int)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

####### Now a RUL VALUE needs to be assigned to each row (cycle) in df_train by subtracting the engine’s current cycle from its max cycle
## By subtracting the max cycle from the current cycle, I can get the remaining useful lifetime of the engine, it yields the number of cycles left before failure - provided the RUL_FD001 vector data

## Calc max cycle per unit
max_cycle_per_unit = dframe_train.groupby('unit')['cycle'].max().reset_index() # used .groupby() to group the data by the unit number, and then used .max() to get the maximum cycle number for each unit, and then reset the index to calc the next engine
max_cycle_per_unit.columns = ['unit', 'max_cycle'] # rename

## merge the max cycle with the RUL data to use it later - essentially appending new and useful initial training data for the model to consider
dframe_rul = dframe_rul.merge(max_cycle_per_unit, on=['unit'], how='left') # used .merge() to merge the RUL data with the max cycle data
                                                                            ## and used 'left' to keep the RUL data as the main data, cycles on left
                                                                            # Also specified on which column, being 'unit' - the unit number


## Calc Rul for each row in training data now
dframe_train = dframe_train.merge(dframe_rul[['unit', 'RUL', 'max_cycle']], on=['unit'], how='left') # merged again here
calculated_RUL_train = dframe_train['max_cycle'] - dframe_train['cycle'] # Max cyc - current cyc yields the RUL value for the training item
    # now update and assign the RUL value for the training item itself
dframe_train['RUL'] = dframe_train['RUL'] + calculated_RUL_train # initial RUL starts at null

dframe_train.drop('max_cycle', axis=1, inplace=True) # just removing the max cycle column after as its not needed (already calculated RUL), can hinder the structure of the dataset later

# print(dframe_train[['unit', 'cycle', 'RUL']].head(10)) # check the new RUL column

with open(personal_fp_train_set, 'r') as f:
    for _ in range(5):  # Print first 5 lines
        print(f.readline())

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ## Briefly testing if the data preprocessed fully
# try:
#     n_display = int(input("Enter # Rows To Display: "))
#     test_display = dframe_train.head(n_display) # .head() will display the inputted amount of rows, or assumes 5 rows as default
    
#     max_row = dframe_train.shape ## .shape (also since its an attribute and not a function there is no .() ) will return a tuple of the dimensions of the dataframe, so I can compare the inputted row quantity to the total number of rows in the dataset, and see if the quantity of rows are exceeded
#                               # The first element of the tuple is the number of rows, and the second element is the number of columns

#     if n_display > max_row[0]:
#         print("The number of rows you entered is greater than the total number of rows in the dataset.")
#     else:
#         print(test_display)
# except ValueError: 
#     print("Row quantities exist only as integers.")
        
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## This project will contain 2 models - one for multi-class classification of failure type to occure in a range of time for the
## engine, including: normalcy, minor fault, major fault, imminent failure. It will also include a regression for predicting the
## RUL (remaining useful-lifetime) of the engine considering all the feautures, data, etc.


## I need to define the return values, and weightage of certain RUL Values


# the RUL data-feature is already saved in the CMAPPS dataset, as ## rul

# def label_RUL_severity(rul): 
#     if rul > 100:
#         return 0  # Normal
#     elif 50 < rul <= 100:
#         return 1  # Minor Fault
#     elif 20 < rul <= 50:
#         return 2  # Major Fault
#     else:
#         return 3  # Imminent Failure
    
# # # apply the comparisons to the dataframe (so return values for each data entry along an axis can be assigned)

# dframe['failure_class'] = dframe['RUL'].apply(label_RUL_severity) # compare now, and apply the function

# # quick check

# distribution_values = dframe['failure_class'].value_counts() ## using .value_counts() helps return a series of unique counts of values in the data axis

# print(distribution_values)

# print("The RUL data-feature is not present in the dataset. Please double check case-sensitivity.")


## Now I will split the data into training and testing sets, and then apply the models

# from sklearn.model_selection import train_test_split
