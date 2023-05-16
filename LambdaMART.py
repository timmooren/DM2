import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import feature_engineering, split, preprocess

df = pd.read_csv("data/training_set_VU_DM.csv")
preprocess(df)

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = split(df)

# Create a LightGBM dataset for training
train_data = lgb.Dataset(X_train, label=y_train)

# Calculate the number of samples per query for the training set
query_counts_train = X_train["srch_id"].value_counts().sort_index().values

# Add query information (group) to the training dataset
train_data.set_group(query_counts_train)


# Specify the configuration for the LightGBM
param = {"objective": "lambdarank", "metric": "ndcg", "ndcg_at": 10}

# Train the model
model = lgb.train(param, train_data, num_boost_round=100)

# Predict the scores for the test set
y_pred = model.predict(X_test, predict_disable_shape_check=True)

# now use y_pred for ranking the hotel room bookings

# Convert predictions to a pandas DataFrame
predictions = pd.DataFrame(data={"prediction": y_pred})

# Join with the test set to get the hotel ids and search ids
ranked_hotels = pd.concat([X_test.reset_index(), predictions], axis=1)

# add the y_test values to the ranked_hotels DataFrame
ranked_hotels["relevance"] = y_test

# Sort/rank hotels by prediction for each search id
ranked_hotels.sort_values(
    by=["srch_id", "prediction"], ascending=[True, False], inplace=True
)

# You now have a DataFrame of hotels in the test set, ranked by likelihood of booking for each search id

# save columns srch_id, prop_id, prediction to csv
ranked_hotels[["srch_id", "prop_id", "relevance", "prediction"]].to_csv(f"ranked_hotels.csv")