# K-Nearest Neighbors (K-NN)

# Importing the libraries
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def main():
    # Importing the dataset (relative to script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = pd.read_csv(os.path.join(script_dir, 'Social_Network_Ads.csv'))
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Visualising the Training set results
    _plot_decision_boundary(
        classifier, X_train, y_train, title='K-NN (Training set)'
    )

    # Visualising the Test set results
    _plot_decision_boundary(
        classifier, X_test, y_test, title='K-NN (Test set)'
    )


def _plot_decision_boundary(classifier, X_set, y_set, title):
    """Plot the decision boundary for a 2D classifier."""
    colors = ('red', 'green')
    plt.figure()
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1,
                  stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1,
                  stop=X_set[:, 1].max() + 1, step=0.01),
    )
    plt.contourf(
        X1, X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(colors),
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c=colors[i],
            label=j,
            edgecolors='black',
            s=25,
        )
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
