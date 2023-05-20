from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.impute import KNNImputer
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
import xgboost as xgb



def feature_engineering(df):
    # add a column that indicates the percental difference in price between that row and the average price of search
    mean_search_price = df.groupby("srch_id")["price_usd"].transform("mean")
    df["price_diff"] = (df["price_usd"] - mean_search_price) / mean_search_price

def normalize(df):
    columns_to_normalize = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2',
       'prop_log_historical_price', 'price_usd', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 
       'srch_children_count', 'srch_room_count', 'orig_destination_distance',  'comp_rate_ratio', 'comp_inv_ratio']
    
    df[columns_to_normalize] = normalize(df[columns_to_normalize])
    
    
def remove_variables(df):
    # variables to remove
    to_drop = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_query_affinity_score']

    # remove variables
    df = df.drop(to_drop, axis=1)
    return df

def data_transformation(df, boxcox=True):
    log_transform = ['prop_location_score1', 'srch_booking_window', 'srch_adults_count', 'srch_room_count', 'orig_destination_distance']
    power_transform = ['prop_location_score2',  'srch_children_count', 'srch_room_count', 'srch_length_of_stay']

    # boxcox transformation
    if boxcox:
        for column in log_transform + power_transform:
            df[column] = df[column] - df[column].min() + 1
            df[column] = pd.Series(boxcox(df[column])[0])
    else:
        for column in log_transform:
            df[column] = np.log(df[column] + 1)
        
        for column in power_transform:
            df[column] = np.power(df[column], 0.5)

def outlier_detection(df, transformed=True):
    # check numerical columns
    columns_to_check = ['prop_location_score1', 'prop_location_score2','prop_log_historical_price', 'price_usd', 'srch_length_of_stay', 'srch_booking_window','srch_adults_count',
                    'srch_children_count', 'srch_room_count', 'orig_destination_distance', 'comp_rate_ratio', 'comp_inv_ratio']
    all_outliers = []
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
        outliers = transformed_column[(transformed_column < lower_bound) | (transformed_column > upper_bound)].index
        all_outliers = list(set(all_outliers + list(outliers)))
        # print(f'{column} has {len(outliers)} outliers')
    return all_outliers

def impute_linear(df):
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
        # print(regression_model.score(training_set[predictors[index]], training_set[dependent]))
        predicted_values = regression_model.predict(missing_set[predictors[index]])

        # replace missing values with predicted values
        if dependent == 'prop_review_score':
            predicted_values = [round(x * 2) / 2 for x in predicted_values.copy()]
        missing_set[dependent] = predicted_values

        # combine dataframes
        df = pd.concat([training_set, missing_set]).sort_index()
    return df

def impute_knn(df):
    dependents = ['orig_destination_distance', 'prop_location_score2', 'prop_review_score']

    # linear imputer
    imputer = KNNImputer(n_neighbors=2)
    imputer.fit(df[dependents])
    df[dependents] = imputer.transform(df[dependents])

def predicted_position(df):
    df = df.drop(columns="date_time")
    df = df.drop(columns="gross_bookings_usd")

    # predict position
    X = df.drop('position', axis=1)
    y = df['position']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert SVMLight format
    dump_svmlight_file(X_train, y_train, 'train.txt')
    dump_svmlight_file(X_test, y_test, 'test.txt')

    dtrain = xgb.DMatrix('train.txt')

    # Set LambdaMART parameters
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@1-5',
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1,
        'alpha': 0,
        'silent': 1,
        'nthread': 4
    }

    # train the LambdaMART model using XGBoost
    model = xgb.train(params, dtrain, num_boost_round=100)

    # make predictions on the test set
    dtest = xgb.DMatrix('test.txt')
    y_pred = model.predict(dtest)

    # add predicted values to df
    df['predicted_position'] = y_pred

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

def categorical_encoding(df, alpha=1):
    '''
    Replaces category values with a weighted average of target encoding average & overall target average. 
    Function assumes that 'relevance' exists. 
    param@alpha: controls the weighted average, (default: alpha=1) meaning that it only uses the target encoding 
    '''
        
    # add prop_id & search_id??
    categorical_features = ['site_id', 'prop_country_id', 'visitor_location_country_id', 
                            'promotion_flag', 'srch_destination_id', 'srch_length_of_stay', 
                            'srch_booking_window', 'srch_adults_count', 'srch_children_count',
                            'srch_room_count', 'srch_saturday_night_bool', 'relevance'] 

    # get categorical features & make new encoded df
    df_copy = df.copy()
    X = df_copy[categorical_features]
    encoded_features = X.copy()

    # overall target average 
    overall_avg = X['relevance'].mean()

    for var in X.drop('relevance', axis=1):
        # get all unique values 
        unique_values = X[var].unique()

        for value in unique_values:
            # get avg feature value per categorical value 
            matching_target_values = X.loc[X[var] == value, 'relevance']
            target_avg = matching_target_values.mean()
            # weighted average 
            weighted_avg = (alpha * target_avg + (1-alpha) * overall_avg) 
            # update encoded feature 
            encoded_features[var].replace(value, weighted_avg, inplace=True)
    
    # replace df with encoded features 
    df_copy = df_copy.drop(categorical_features, axis=1)
    df_copy[categorical_features] = encoded_features

    return df_copy

def down_sampling(df, class_val=0, alpha=0.5, with_replacement=False):
    '''
    Down samples the specified class value to specified percentage of instances. 
    Function assumes that 'relevance' exists. 
    param@class_val: value of the class to downsample (0,1,6)
    param@alpha: percentage of selected class instances to downsample (1=all, 0=none). 
    '''
    df_copy = df.copy()

    # print percentages of class labels before downsampling
    temp = ''
    for val in (0,1,6):
        val_rate = df_copy['relevance'].value_counts()[val] / len(df_copy)
        temp += f'{val}: {val_rate}'
    print(f'before downsampling {temp}')

    # nb of samples to retain 
    n_samples = round(df_copy['relevance'].value_counts()[class_val] * alpha)
    # create new downsampled df  
    downsampled_df = df_copy[df_copy['relevance'] == class_val].sample(n_samples, replace=with_replacement)
    remaining_df = df_copy[df_copy['relevance'] != class_val]
    result_df = pd.concat([downsampled_df, remaining_df])

    # print percentage of class labels after downsampling  
    temp = ''
    for val in (0,1,6):
        val_rate = result_df['relevance'].value_counts()[val] / len(result_df)
        temp += f'{val}:{round(val_rate, 3)} '
    print(f'after downsampling {temp}')

    return result_df


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
