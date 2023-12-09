#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# # Load the data

# In[2]:


df_train = pd.read_parquet('train.parquet')
df_test = pd.read_parquet('final_test.parquet')
df_external = pd.read_csv('external_data.csv')
df_external = df_external.drop_duplicates()


# # Feature Engineering

# Perform feature engineering on the original train data

# In[3]:


# Perform feature engineering on the original train data

# Temporal Features
df_train['hour'] = df_train['date'].dt.hour
df_train['day_of_week'] = df_train['date'].dt.dayofweek
df_train['month'] = df_train['date'].dt.month
df_train['year'] = df_train['date'].dt.year
df_train['is_weekend'] = df_train['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Counter Age
current_date = df_train['date'].max()  # Using the latest date in the dataset as the current date
df_train['counter_age'] = (current_date - df_train['counter_installation_date']).dt.days

# Categorical Encoding (using one-hot encoding as an example)
# We'll encode 'day_of_week' and 'hour' as categorical features for this example
df_train = pd.get_dummies(df_train, columns=['day_of_week', 'hour'])

# Geospatial Features (example: distance to a hypothetical city center at coordinates (48.8566, 2.3522))
from geopy.distance import geodesic

# Define a function to calculate the distance to the city center
def calculate_distance_to_center(row):
    city_center_coords = (48.8566, 2.3522)
    counter_coords = tuple(map(float, row['coordinates'].split(',')))
    return geodesic(counter_coords, city_center_coords).kilometers

# Apply the function to each row
df_train['distance_to_center'] = df_train.apply(calculate_distance_to_center, axis=1)

# Display the first few rows with the new features
# df_merged[['counter_age', 'is_weekend', 'distance_to_center'] + [col for col in df_merged.columns if 'day_of_week' in col or 'hour_' in col]].head()


# Weather Feature, processed directly on the external data

# In[4]:


df_external['date'] = pd.to_datetime(df_external['date'])


# Temperature (t) & Temperature Binning
df_external['t_Celsius'] = df_external['t'] - 273.15 # change to celsius
df_external['t_Celsius'].apply(lambda x: 0 if pd.isna(x) else x)

temperature_bins = [-np.inf, 0, 15, 28, np.inf]
temperature_bin_labels = ['cold', 'cool', 'warm', 'hot']

df_external['temp_bin'] = pd.cut(df_external['t'], bins=temperature_bins, labels=temperature_bin_labels)

# Precipitation (rr24) & Precipitation Binning
df_external['rr24_corrected'] = df_external['rr24'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x) # correct negative and nan values

precipitation_bins = [-np.inf, 0, 10, np.inf]
precipitation_bin_labels = ['dry', 'drizzling', 'rainy']

df_external['precipitation_bin'] = pd.cut(df_external['rr24_corrected'], bins=precipitation_bins, labels=precipitation_bin_labels)


# Wind Speed (ff) & Wind Speed Binning
wind_speed_bins = [-np.inf, 4, 7, np.inf]
wind_labels = ['calm', 'moderate', 'strong']
df_external['wind_bin'] = pd.cut(df_external['ff'], bins=wind_speed_bins, labels=wind_labels)

# Visibility (vv) & Binary indicator (1: good visibility; 0: bad visibility)
df_external['vv_indicator'] = df_external['vv'].apply(lambda x: 1 if x > 6000 else 0)

# Total Cloudiness (n) & Cloudiness Binning
cloudiness_bins = [-np.inf, 20, 75, np.inf]
cloudiness_labels = ['Sunny', 'Moderate', 'Cloudy']
df_external['n_bin'] = pd.cut(df_external['n'], bins=cloudiness_bins, labels=cloudiness_labels)


# Perform feature engineering on the test data

# In[5]:


# Perform feature engineering on the test data

# Temporal Features
df_test['hour'] = df_test['date'].dt.hour
df_test['day_of_week'] = df_test['date'].dt.dayofweek
df_test['month'] = df_test['date'].dt.month
df_test['year'] = df_test['date'].dt.year
df_test['is_weekend'] = df_test['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Counter Age
current_date = df_test['date'].max()  # Using the latest date in the dataset as the current date
df_test['counter_age'] = (current_date - df_test['counter_installation_date']).dt.days

# Categorical Encoding (using one-hot encoding as an example)
# We'll encode 'day_of_week' and 'hour' as categorical features for this example
df_test = pd.get_dummies(df_test, columns=['day_of_week', 'hour'])

# Geospatial Features (example: distance to a hypothetical city center at coordinates (48.8566, 2.3522))
from geopy.distance import geodesic

# Define a function to calculate the distance to the city center
def calculate_distance_to_center(row):
    city_center_coords = (48.8566, 2.3522)
    counter_coords = tuple(map(float, row['coordinates'].split(',')))
    return geodesic(counter_coords, city_center_coords).kilometers

# Apply the function to each row
df_test['distance_to_center'] = df_test.apply(calculate_distance_to_center, axis=1)

# Display the first few rows with the new features
# df_test_merged[['counter_age', 'is_weekend', 'distance_to_center'] + [col for col in df_test_merged.columns if 'day_of_week' in col or 'hour_' in col]].head()


# Data Merger

# In[6]:


df_merged = df_train.merge(df_external, on='date', how='left')
df_test_merged = df_test.merge(df_external, on='date', how='left')


# In[7]:


df_merged['log_bike_count_corrected'] = df_merged['log_bike_count'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)


# In[8]:


# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# df_merged_cleaned = df_merged.dropna()

# X_cleaned = df_merged_cleaned.drop('log_bike_count', axis=1)

# #print(X_cleaned.columns[20:])


# # XG Boost

# In[9]:


import xgboost as xgb



selected_columns_1 = ['day_of_week_0', 'day_of_week_1', 'day_of_week_2',
       'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
       'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
       'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12',
       'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18',
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 
        'month', 'year', 'is_weekend', 'counter_age', 'distance_to_center']





X = df_merged[selected_columns_1]
y = df_merged['log_bike_count_corrected']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Create an XGBoost regressor model
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, seed=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

# print("XGBoost RMSE: ", rmse_xgb)

# Feature Importance
feature_importances_xgb = xgb_model.feature_importances_

print("XGBoost RMSE: ", rmse_xgb)


# Implement GridSearch method to estimate the best parameters

# In[10]:


param_grid = {
    'max_depth': [4, 6, 7],
    'min_child_weight': [1, 2],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.9, 1.0, 1.1],
    'colsample_bytree': [0.9, 1.0, 1.1],
    'learning_rate': [0.2, 0.3, 0.4]
}


from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb.XGBRegressor(objective ='reg:squarederror', seed=42), 
                           param_grid, 
                           cv=3, 
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)


grid_search.fit(X_train, y_train)


print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))


# In[11]:


xgb_model_best = xgb.XGBRegressor(
    objective ='reg:squarederror',
    colsample_bytree=1.0,
    gamma=0,
    learning_rate=0.4,
    max_depth=7,
    min_child_weight=2,
    subsample=1.0,
    n_estimators=100, 
    seed=42           
)


xgb_model_best.fit(X_train, y_train)


y_pred_xgb_best = xgb_model_best.predict(X_test)


mse_xgb_best = mean_squared_error(y_test, y_pred_xgb_best)
rmse_xgb_best = np.sqrt(mse_xgb_best)

print("XGBoost RMSE with Best Parameters: ", rmse_xgb_best)


# In[12]:


y_pred_xgb_best = xgb_model_best.predict(df_test_merged[selected_columns_1])
results_xgb_best = pd.DataFrame(
    dict(
        Id=np.arange(y_pred_xgb_best.shape[0]),
        log_bike_count=y_pred_xgb_best,
    )
)
results_xgb_best.to_csv("submission_xgb_best.csv", index=False)


# In[ ]:




