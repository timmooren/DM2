from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split


def train_model(X, y):
    # HistGradientBoostingClassifier is a gradient boosting model that uses histogram-based optimizations.
    model = HistGradientBoostingClassifier()
    model.fit(X, y)
    return model


def predict_class(model, sample):
    # Predict the class of the given sample
    prediction = model.predict(sample)

    # Print the predicted class
    print(prediction)


def main():
    df = pd.read_csv("data/training_set_VU_DM.csv", nrows=1000)

    # train test split
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # drop columns
    X_train = train.drop(
        columns=[
            "date_time",
            "srch_id",
            "prop_id",
            "click_bool",
            "booking_bool",
            "gross_bookings_usd",
        ]
    )
    y_train = train["booking_bool"]

    # train model
    model = train_model(X_train, y_train)

    # test model
    X_test = test.drop(
        columns=[
            "date_time",
            "srch_id",
            "prop_id",
            "click_bool",
            "booking_bool",
            "gross_bookings_usd",
        ]
    )
    predict_class(
        model, X_test.iloc[[0]]
    )  # Note the double brackets to keep DataFrame format


if __name__ == "__main__":
    main()
