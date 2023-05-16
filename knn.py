from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import feature_engineering, split, ndcg_score, preprocess


def train_model(X, y):
    # ball tree is a type of data structure that can be used to efficiently find the nearest neighbors in a multi-dimensional space.
    model = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
    model.fit(X, y)
    return model


def nearest_neighbors(model, hotel, y_true):
    distances, indices = model.kneighbors(hotel)
    y_pred = y_true[indices.ravel()]  # Get the predicted rankings

    ndcg = ndcg_score(y_true, y_pred)
    print(f"NDCG Score: {ndcg}")


def main():
    df = pd.read_csv("data/training_set_VU_DM.csv", nrows=1000)
    preprocess(df)

    # train test split
    X_train, X_test, y_train, y_test = split(df)

    # train model
    model, y_train = train_model(X_train, y_train)

    # test model
    nearest_neighbors(model, X_train.iloc[0].values.reshape(1, -1), y_train)


if __name__ == "__main__":
    main()
