import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df_jan22 = pd.read_parquet("yellow_tripdata_2022-01.parquet")
df_feb22 = pd.read_parquet("yellow_tripdata_2022-02.parquet")

# print(df_jan22.info)
# print(df_jan22.head)
# print(df_jan22.columns)
print(f"Question 1: # columns {len(df_jan22.columns)}")

# Question 2: What is the std dev of the Jan 2022 Trips
df_jan22['duration'] = (df_jan22['tpep_dropoff_datetime'] - df_jan22['tpep_pickup_datetime'])
df_jan22['duration_mins'] = df_jan22['duration'].dt.total_seconds()/60

# print(df_jan22.head)
print(df_jan22.describe())


# Question 3: dropping outliers
df_jan22['outlier_duration'] = np.where((df_jan22['duration_mins'] > 60) | (df_jan22['duration_mins'] < 1), 1, 0)
outlier_rate = np.sum(df_jan22['outlier_duration'])/len(df_jan22['outlier_duration'])
print(f"Question 3: Outlier Rate {(1-outlier_rate)*100}")
df_jan22_clean = df_jan22[df_jan22['outlier_duration'] == 0]


# Question 4: OHE

# https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit#heading=h.y8azg3q0vqek
jan_dict = df_jan22_clean[['DOLocationID', 'PULocationID']].fillna(-1).astype('int').astype('str').to_dict('records')
dv = DictVectorizer()
X_train = dv.fit_transform(jan_dict)
print(f"Question 4: Number of features {len(dv.feature_names_)}")

# Question 5: RSME on train
y_train = df_jan22_clean['duration_mins'].values
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X_train, y_train)  # perform linear regression
Y_pred = linear_regressor.predict(X_train)  # make predictions
rsme = mean_squared_error(y_train, Y_pred, squared=False)
print(f"Question 5: Train rsme {rsme}")


# Question 6: RSME on test
# mean_squared_error(y_true, y_pred, squared=False)
# todo: this would be better as a function
df_feb22['duration'] = (df_feb22['tpep_dropoff_datetime'] - df_feb22['tpep_pickup_datetime'])
df_feb22['duration_mins'] = df_feb22['duration'].dt.total_seconds()/60
df_feb22['outlier_duration'] = np.where((df_feb22['duration_mins'] > 60) | (df_feb22['duration_mins'] < 1), 1, 0)
df_feb22_clean = df_feb22[df_feb22['outlier_duration'] == 0]
feb_dict = df_feb22_clean[['DOLocationID', 'PULocationID']].fillna(-1).astype('int').astype('str').to_dict('records')
X_test = dv.transform(feb_dict)
Y_test = df_feb22_clean['duration_mins'].values
Y_pred = linear_regressor.predict(X_test)
rsme = mean_squared_error(Y_test, Y_pred, squared=False)
print(f"Question 6: Test rsme {rsme}")

