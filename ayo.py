import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def assign_season(date):
    month = int(str(date).split("-")[1])
    if month in [6, 7, 8, 9]:
        return 2    # Rainy
    elif month in [11, 12, 1, 2]:
        return 1    # Winter
    else:
        return 0    # Summer

# Load data
data1 = pd.read_csv("dengue_features_train.csv")
data2 = pd.read_csv("dengue_labels_train.csv")

# Replace blank or whitespace-only strings with NaN
data1 = data1.replace(r'^\s*$', np.nan, regex=True)
data2 = data2.replace(r'^\s*$', np.nan, regex=True)

# Convert columns that should be numeric (example: temperature, rainfall, etc.)
numeric_cols = data1.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Fill NaN only in numeric columns using their mean
data1[numeric_cols] = data1[numeric_cols].fillna(data1[numeric_cols].mean())

# Similarly for data2 (assuming it has numeric columns)
numeric_cols2 = data2.select_dtypes(include=['float64', 'int64']).columns.tolist()
data2[numeric_cols2] = data2[numeric_cols2].fillna(data2[numeric_cols2].mean())

# Map city names to integers
data1["city"] = data1["city"].map({"sj": 0, "iq": 1})

# Create a new 'season' column without overwriting the original date
data1['week_start_date'] = data1['week_start_date'].apply(assign_season)

#train the model
x_train=data1
y_train=data2["total_cases"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#test the model
data3=pd.read_csv("dengue_features_test.csv")

#clean the data
data3 = data3.replace(r'^\s*$', np.nan, regex=True)

#convert the nan
numeric_cols3 = data3.select_dtypes(include=['float64', 'int64']).columns.tolist()
data3[numeric_cols3] = data3[numeric_cols3].fillna(data3[numeric_cols3].mean())
data3["city"] = data3["city"].map({"sj": 0, "iq": 1})
data3['week_start_date'] = data3['week_start_date'].apply(assign_season)

#time to test
x_test=data3
predictions = model.predict(x_test)
case_counts = np.round(predictions).astype(int)

#writing the final data
data4=pd.read_csv("submission_format.csv")
data4['total_cases'] = case_counts

data4.to_csv("output.csv",index=False)