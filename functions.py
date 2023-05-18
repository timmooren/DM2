from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


def feature_engineering(df):
    # add a column that indicates the percental difference in price between that row and the average price of search
    mean_search_price = df.groupby("srch_id")["price_usd"].transform("mean")
    df["price_diff"] = (df["price_usd"] - mean_search_price) / mean_search_price

    
def remove_variables(df):
    # variables to remove
    to_drop = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_query_affinity_score']

    # remove variables
    df = df.drop(to_drop, axis=1)
    return df

def impute_missing_values(df):
    # predictor and dependent variables
    dependents = ['orig_destination_distance', 'prop_location_score2', 'prop_review_score']
    predictors = [['prop_country_id', 'visitor_location_country_id', 'srch_destination_id'], ['prop_location_score1', 'prop_id'], ['prop_starrating', 'prop_id', 'prop_location_score1']]

    # loop over all dependent variables
    for dependent in dependents:
        index = dependents.index(dependent)
        # split data into train and test sets
        training_set = df.dropna(subset=[dependent])
        missing_set = df[df[dependent].isnull()]

        # linear regression model
        regression_model = LinearRegression()
        regression_model.fit(training_set[predictors[index]], training_set[dependent])

        predicted_values = regression_model.predict(missing_set[predictors[index]])

        # replace missing values with predicted values
        if dependent == 'prop_review_score':
            predicted_values = [round(x * 2) / 2 for x in predicted_values.copy()]
        missing_set[dependent] = predicted_values

        # combine dataframes
        df = pd.concat([training_set, missing_set]).sort_index()
    return df

def comp_aggregation(df):
    df_copy = df.copy()
    # competitor rate variables 
    comp_rate_vars = ['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']
    df_copy['comp_rate_ratio'] = df_copy[comp_rate_vars].mean(axis=1)
    # replace NaN with 0 & remove original vars
    df_copy['comp_rate_ratio'] = df_copy['comp_rate_ratio'].fillna(0)
    df_copy = df_copy.drop(comp_rate_vars, axis=1)


    # competitor availability variables 
    comp_inv_vars = ['comp1_inv', 'comp2_inv', 'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']
    df_copy['comp_inv_ratio'] = df_copy[comp_inv_vars].mean(axis=1)
    # replace NaN with 0.5 & remove original vars
    df_copy['comp_inv_ratio'] = df_copy['comp_rate_ratio'].fillna(0.5)
    df_copy = df_copy.drop(comp_inv_vars, axis=1)

    # remove competitor rate difference because we said so 
    comp_rate_diff_vars = ['comp1_rate_percent_diff', 'comp2_rate_percent_diff', 'comp3_rate_percent_diff', 
                           'comp4_rate_percent_diff', 'comp5_rate_percent_diff', 'comp6_rate_percent_diff', 
                           'comp7_rate_percent_diff', 'comp8_rate_percent_diff']
    df_copy = df_copy.drop(comp_rate_diff_vars, axis=1)

    return df_copy


def preprocess(df):
    df["relevance"] = df["booking_bool"].apply(lambda x: 5 if x == 1 else 0) + df[
        "click_bool"
    ].apply(lambda x: 1 if x == 1 else 0)
    # drop datetime
    df.drop(columns="date_time", inplace=True)
    # save
    df.to_csv("data/preprocessed.csv", index=False)


def split(df):
    # train test split
    to_drop = [
        "click_bool",
        "booking_bool",
        "gross_bookings_usd",
        "position",
    ]
    X = df.drop(columns="relevance")

    X_train, X_test, y_train, y_test = train_test_split(
        X, df["relevance"], test_size=0.2, random_state=42
    )

    # drop columns from test
    X_test.drop(columns=to_drop, inplace=True)

    return X_train, X_test, y_train, y_test


def ndcg_score(y_true, y_pred):
    # NDCG requires the relevance scores to be in the range [0, 1]
    y_true = np.clip(y_true, 0, 1)
    y_pred = np.clip(y_pred, 0, 1)
    return ndcg_score([y_true], [y_pred])
