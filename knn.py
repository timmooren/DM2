from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import feature_engineering, split


def train_model(X, y):
    # ball tree is a type of data structure that can be used to efficiently find the nearest neighbors in a multi-dimensional space.
    model = NearestNeighbors(n_neighbors=5, algorithm="ball_tree")
    model.fit(
        X,
    )
    return model


def nearest_neighbors(model, hotel):
    # Find the k nearest neighbors
    distances, indices = model.kneighbors(hotel)

    # The indices are the ranks of the hotel rooms for the given user
    print(indices)


def main():
    df = pd.read_csv("data/training_set_VU_DM.csv", nrows=1000)

    X, y = split(df)


    # train model
    model = train_model(X, y)

    # test model
    nearest_neighbors(model, X.iloc[0].values.reshape(1, -1))


if __name__ == "__main__":
    main()
