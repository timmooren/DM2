from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
import xgboost as xgb
import pickle
import json

categorical_variables = [
    "month",
    "hour",
    "dayofweek",
    "site_id",
    "prop_id",
    "prop_country_id",
    "visitor_location_country_id",
    "promotion_flag",
    "srch_destination_id",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "random_bool",
]


def all_steps(df, test=False, knn=False):
    print("preprocessing")
    df = preprocess(df, test=test)  # add relevance column
    print("remove variables")
    df = remove_variables(df, test=test)
    print("aggregation")
    df = comp_aggregation(df)  # feature engineering
    print("remove outliers")
    if not test:
        df = remove_outliers(df)
    df = date_features(df)  # feature engineering

    print("imputing")
    df = impute_linear(df)

    # print nan values
    # df = impute_knn(df)
    print("percental difference")
    df = percental_difference(df)  # feature engineering

    # print nan values
    print("cat encoding")
    if not test:
        df = categorical_encoding_train(df)
    else:
        df = categorical_encoding_test(df)
        # print nan values

    # print("data transformation")
    # df = data_transformation(df)
    if knn:
        df = normalize(df)

    print("down sampling")
    # df = down_sampling(df)  # or weighted something
    print("predict position")
    df = predicted_position(df, test=test)
    # print nan values

    # assert there are no missing values in df
    assert df.isnull().sum().sum() == 0, "There are missing values in the dataframe"
    return df


def date_features(df, col="date_time", features=["month", "hour", "dayofweek"]):
    """Extracts date features from a column in a dataframe and adds them to the dataframe."""
    print(f"Extracting date features from {col}...")
    dates = pd.to_datetime(df[col])
    for feature in features:
        if feature == "hour":
            df["hour"] = dates.dt.hour
        elif feature == "dayofweek":
            df["dayofweek"] = dates.dt.dayofweek
        elif feature == "month":
            df["month"] = dates.dt.month
    # drop original column
    df.drop(col, axis=1, inplace=True)
    return df


def percental_difference(
    df,
    variables=[
        "price_usd",
        "prop_location_score1",
        "prop_location_score2",
        "prop_review_score",
        "prop_starrating",
    ],
):
    # add a column that indicates the percental difference in price between that row and the average price of search
    for variable in variables:
        mean = df.groupby("srch_id")[variable].transform("mean")
        df[f"{variable}_diff"] = (df[variable] - mean) / mean
        # replace missing values with overall mean for that variable
        df[f"{variable}_diff"] = df[f"{variable}_diff"].fillna(
            df[f"{variable}_diff"].mean()
        )
        # add colomn for standard deviation for variable
        df[f"{variable}_std"] = df.groupby("srch_id")[variable].transform("std")
        # replace missing values with overall mean for that variable
        df[f"{variable}_std"] = df[f"{variable}_std"].fillna(
            df[f"{variable}_std"].mean()
        )
        # add column for median for variable
        df[f"{variable}_median"] = df.groupby("srch_id")[variable].transform("median")
        # replace missing values with overall mean for that variable
        df[f"{variable}_median"] = df[f"{variable}_median"].fillna(
            df[f"{variable}_median"].mean()
        )

    return df


def normalize(df):
    scaler = preprocessing.MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    return df


def remove_variables(df, test=False):
    # variables to remove
    to_drop = [
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "srch_query_affinity_score",
    ]
    if not test:
        to_drop.extend(["click_bool", "booking_bool", "gross_bookings_usd"])

    # for column in to_drop:
    #     if column in df.columns:
    #         df.drop(columns=[column], inplace=True)
    df.drop(columns=to_drop, inplace=True)

    return df


def data_transformation(df, use_boxcox=True):
    log_transform = [
        "prop_location_score1",
        "srch_booking_window",
        "srch_adults_count",
        "srch_room_count",
        "orig_destination_distance",
    ]
    power_transform = [
        "prop_location_score2",
        "srch_children_count",
        "srch_room_count",
        "srch_length_of_stay",
    ]

    # boxcox transformation
    if use_boxcox:
        for column in log_transform + power_transform:
            df[column] = df[column] - df[column].min() + 0.01
            df[column] = pd.Series(boxcox(df[column])[0])
    else:
        for column in log_transform:
            df[column] = np.log(df[column] + 1)

        for column in power_transform:
            df[column] = np.power(df[column], 0.5)
    return df


def outlier_detection(df, transformed=False):
    # check numerical columns
    columns_to_check = [
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "price_usd",
        "srch_length_of_stay",
        "srch_booking_window",
        "orig_destination_distance",
        "comp_rate_ratio",
        "comp_inv_ratio",
    ]
    all_outliers = {}
    # detect outliers for every column
    for column in columns_to_check:
        transformed_column = df[column]

        if not transformed:
            # make negative values positive
            df[column] = df[column] - df[column].min() + 1
            transformed_column = pd.Series(boxcox(df[column])[0])

        # use iqr to remove outliers
        Q1 = transformed_column.quantile(0.25)
        Q3 = transformed_column.quantile(0.75)
        IQR = Q3 - Q1
        # use 3 for extreme outliers
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        # get index of outliers in transformed
        outliers = transformed_column[
            (transformed_column < lower_bound) | (transformed_column > upper_bound)
        ].index
        all_outliers[column] = list(outliers)
        # all_outliers = list(set(all_outliers + list(outliers)))
        # print(f'{column} has {len(outliers)} outliers')
    return all_outliers


def remove_outliers(df):
    all_outliers = outlier_detection(df)
    # remove outliers of price_usd
    price_outliers = all_outliers["price_usd"]
    # drop outliers from index
    df = df.drop(price_outliers, axis=0)
    return df


def zero_to_null(df, variables):
    # replace 0s with NaN for given columns
    df[variables] = df[variables].replace(0, np.nan)
    # print(df.isnull().sum())
    return df


def impute_linear(df):
    # predictor and dependent variables
    df = zero_to_null(
        df, ["prop_review_score", "prop_starrating", "prop_log_historical_price"]
    )

    # make a list of columns with nan values
    dependents = df.columns[df.isna().any()].tolist()
    predictors = list(set(df.columns) - set(dependents) - set(["position"]))

    # loop over all dependent variables
    for dependent in dependents:
        # split data into train and test sets
        training_set = df.dropna(subset=[dependent])
        missing_set = df[df[dependent].isnull()]

        # linear regression model
        regression_model = LinearRegression()
        regression_model.fit(training_set[predictors], training_set[dependent])

        # print(regression_model.score(training_set[predictors[index]], training_set[dependent]))
        predicted_values = regression_model.predict(missing_set[predictors])

        # replace missing values with predicted values
        missing_set[dependent] = predicted_values

        # combine dataframes
        df = pd.concat([training_set, missing_set]).sort_index()
    return df


def impute_knn(df):
    # TODO: NOT WORKING
    dependents = [
        "orig_destination_distance",
        "prop_location_score2",
        "prop_review_score",
        "prop_starrating",
        "prop_log_historical_price",
    ]
    df = zero_to_null(
        df, ["prop_review_score", "prop_starrating", "prop_log_historical_price"]
    )
    #

    # simple imputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df[dependents])
    df[dependents] = imputer.transform(df[dependents])
    return df


def predicted_position(df, test=False):
    # df = df.drop(columns="date_time")
    # df = df.drop(columns="gross_bookings_usd")

    if not test:
        # predict position
        X = df.drop("position", axis=1)
        y = df["position"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Convert SVMLight format
        dump_svmlight_file(X_train, y_train, "train.txt")
        dump_svmlight_file(X_test, y_test, "test.txt")

        dtrain = xgb.DMatrix("train.txt")

        # Set LambdaMART parameters
        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@1-5",
            "eta": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "gamma": 0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1,
            "alpha": 0,
            "nthread": 4,
        }

        # train the LambdaMART model using XGBoost
        model = xgb.train(params, dtrain, num_boost_round=100)
        # save model using pickle
        pickle.dump(model, open("model.pkl", "wb"))
        df.drop(columns="position", inplace=True)

    else:
        X = df
        # load model from model.pkl
        model = pickle.load(open("model.pkl", "rb"))

    # make predictions on the test set
    # dtest = xgb.DMatrix("test.txt")
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)

    # add predicted values to df
    df["predicted_position"] = y_pred

    return df


def comp_aggregation(df):
    # competitor rate variables
    comp_rate_vars = [
        "comp1_rate",
        "comp2_rate",
        "comp3_rate",
        "comp4_rate",
        "comp5_rate",
        "comp6_rate",
        "comp7_rate",
        "comp8_rate",
    ]
    df["comp_rate_ratio"] = df[comp_rate_vars].mean(axis=1)
    # replace NaN with 0 & remove original vars
    df["comp_rate_ratio"] = df["comp_rate_ratio"].fillna(0)
    df.drop(comp_rate_vars, axis=1, inplace=True)

    # competitor availability variables
    comp_inv_vars = [
        "comp1_inv",
        "comp2_inv",
        "comp3_inv",
        "comp4_inv",
        "comp5_inv",
        "comp6_inv",
        "comp7_inv",
        "comp8_inv",
    ]
    df["comp_inv_ratio"] = df[comp_inv_vars].mean(axis=1)
    # replace NaN with 0.5 & remove original vars
    df["comp_inv_ratio"] = df["comp_inv_ratio"].fillna(0.5)
    df.drop(comp_inv_vars, axis=1, inplace=True)

    # remove competitor rate difference because we said so
    comp_rate_diff_vars = [
        "comp1_rate_percent_diff",
        "comp2_rate_percent_diff",
        "comp3_rate_percent_diff",
        "comp4_rate_percent_diff",
        "comp5_rate_percent_diff",
        "comp6_rate_percent_diff",
        "comp7_rate_percent_diff",
        "comp8_rate_percent_diff",
    ]
    df.drop(comp_rate_diff_vars, axis=1, inplace=True)

    return df


def categorical_encoding_ryo(df, alpha=1):
    """
    Replaces category values with a weighted average of target encoding average & overall target average.
    Function assumes that 'relevance' exists.
    param@alpha: controls the weighted average, (default: alpha=1) meaning that it only uses the target encoding
    """

    # add prop_id & search_id??
    # maybe remove srch_length_of_stay, 'srch_adults_count', 'srch_children_count','srch_room_count'?? (more ordinal)
    #
    categorical_features = [
        "month",
        "hour",
        "dayofweek",
        "site_id",
        "prop_country_id",
        "visitor_location_country_id",
        "promotion_flag",
        "srch_destination_id",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_saturday_night_bool",
        "random_bool",
        "relevance",
    ]

    # get categorical features & make new encoded df
    df_copy = df.copy()
    X = df_copy[categorical_features]
    encoded_features = X.copy()

    # overall target average
    overall_avg = X["relevance"].mean()

    for var in X.drop("relevance", axis=1):
        # get all unique values
        unique_values = X[var].unique()

        for value in unique_values:
            # get avg feature value per categorical value
            matching_target_values = X.loc[X[var] == value, "relevance"]
            target_avg = matching_target_values.mean()
            # weighted average
            weighted_avg = alpha * target_avg + (1 - alpha) * overall_avg
            # update encoded feature
            encoded_features[var].replace(value, weighted_avg, inplace=True)

    # replace df with encoded features
    df_copy = df_copy.drop(categorical_features, axis=1)
    df_copy[categorical_features] = encoded_features

    # convert all features to float
    df_copy = df_copy.astype("float64")

    return df_copy


def categorical_encoding_train(df, alpha=1):
    """
    Replaces category values with a weighted average of target encoding average & overall target average.
    Function assumes that 'relevance' exists.
    param@alpha: controls the weighted average, (default: alpha=1) meaning that it only uses the target encoding
    """

    overall_avg = df["relevance"].mean()

    mappings = {}

    for var in [c for c in categorical_variables if c != "relevance"]:
        target_avg_df = (
            df.groupby(var)["relevance"]
            .mean()
            .reset_index()
            .rename(columns={"relevance": f"{var}_target_avg"})
        )
        df = df.merge(target_avg_df, on=var, how="left")
        # df[f"{var}_target_avg"].fillna(overall_avg, inplace=True)
        df[var] = alpha * df[f"{var}_target_avg"] + (1 - alpha) * overall_avg
        mappings[var] = (
            # target_avg_df.set_index(var)[f"{var}_target_avg"].reindex(df[var]).to_dict()
            {row[0]: row[1] for row in zip(target_avg_df[var], df[f"{var}_target_avg"])}
        )
        # df.set_index(var)[f"{var}_target_avg"].to_dict()
        df.drop(columns=[f"{var}_target_avg"], inplace=True)

    df = df.astype("float64")

    # Saving the mappings as a JSON file
    with open("mappings.json", "w") as f:
        json.dump(mappings, f)

    return df


def categorical_encoding_test(df):
    """
    Applies the learned mappings from the training data on the test data.
    """
    # Loading the mappings from the JSON file
    with open("mappings.json") as f:
        mappings = json.load(f)

    for var in [c for c in categorical_variables if c != "relevance"]:
        mean = df[var].mean()
        for row in df[var]:
            # replace with mapping, else mean
            row = mappings[var].get(row, mean)

    df = df.astype("float64")

    return df


def down_sampling(df, class_val=0, alpha=0.05, with_replacement=False):
    # potential problem: making sure that each srch_id has relatively the same amount of relevance labels
    """
    Down samples the specified class value to specified percentage of instances.
    Function assumes that 'relevance' exists.
    param@class_val: value of the class to downsample (0,1,6)
    param@alpha: percentage of selected class instances to downsample (1=all, 0=none).
    """
    df_copy = df.copy()

    # print percentages of class labels before downsampling
    temp = ""
    for val in (0, 1, 5):
        val_rate = df_copy["relevance"].value_counts()[val] / len(df_copy)
        temp += f"{val}: {val_rate}"
    print(f"before downsampling {temp}")

    # nb of samples to retain
    n_samples = round(df_copy["relevance"].value_counts()[class_val] * alpha)
    # create new downsampled df
    downsampled_df = df_copy[df_copy["relevance"] == class_val].sample(
        n_samples, replace=with_replacement
    )
    remaining_df = df_copy[df_copy["relevance"] != class_val]
    result_df = pd.concat([downsampled_df, remaining_df])

    # print percentage of class labels after downsampling
    temp = ""
    for val in (0, 1, 5):
        val_rate = result_df["relevance"].value_counts()[val] / len(result_df)
        temp += f"{val}:{round(val_rate, 3)} "
    print(f"after downsampling {temp}")

    return result_df


def preprocess(df, test=False):
    # # Convert categorical columns to 'category' type
    # for col in categorical_variables:
    #     if col in df.columns:
    #         df[col] = df[col].astype("category")
    if not test:
        df["relevance"] = df["booking_bool"].apply(lambda x: 5 if x == 1 else 0) + df[
            "click_bool"
        ].apply(lambda x: 1 if x == 1 else 0)
        # replace 6 with 5
        df["relevance"] = df["relevance"].apply(lambda x: 5 if x == 6 else x)
    return df

from sklearn.model_selection import GroupShuffleSplit

def split(df, groups=True):
    X = df.drop(columns="relevance")
    y = df['relevance']

    # split based on search_id groups

    # Define a GroupShuffleSplit splitter
    gss = GroupShuffleSplit(n_splits=1, train_size=.8)

    # Split using GroupShuffleSplit
    train_idx, test_idx = next(gss.split(X, y, groups=df['srch_id']))

    # Create train and test dataframes
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def get_categorical_column(df):
    categorical_variables = [
        "month",
        "hour",
        "dayofweek",
        "site_id",
        "prop_id",
        "prop_country_id",
        "visitor_location_country_id",
        "promotion_flag",
        "srch_destination_id",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_saturday_night_bool",
        "random_bool",
    ]
    # returns column numbers of categorical variables
    categorical_variables = [c for c in categorical_variables if c in df.columns.values]
    categorical_features_numbers = [
        df.columns.get_loc(var) for var in categorical_variables
    ]

    return categorical_features_numbers
