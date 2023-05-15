from sklearn.model_selection import train_test_split
import numpy as np

def feature_engineering(df):
    # add a column that indicates the percental difference in price between that row and the average price of search
    mean_search_price = df.groupby("srch_id")["price_usd"].transform("mean")
    df["price_diff"] = (df["price_usd"] - mean_search_price) / mean_search_price


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
