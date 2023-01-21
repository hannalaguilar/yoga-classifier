import time
from typing import Union
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

from yoga import definitions


def grid_search_wrapper(clf,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        tuned_parameters: Union[list, dict],
                        refit_strategy='accuracy'):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid_search = GridSearchCV(clf,
                               tuned_parameters,
                               cv=skf,
                               refit=refit_strategy)
    grid_search.fit(X_train, y_train)
    print(f'------Analysis orientated to {refit_strategy}------:')
    print(f'Best params ({refit_strategy}): {grid_search.best_params_}')
    print(f'Best score ({refit_strategy}): {grid_search.best_score_}')
    return grid_search


def get_X_y_names(data_path: Path):
    df = pd.read_csv(data_path, index_col=0)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    names = df.iloc[:, 0]
    names = [name.split('.')[0] for name in names]
    return X, y, names


def svm_classifiers(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series):
    # Train
    params = [{'kernel': ['rbf'],
               'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1,
                         1, 10, 100],
               'C': [0.1, 1, 5, 10, 15, 50, 100, 250, 500, 1000]},
              {'kernel': ['linear'],
               'C': [0.1, 1, 5, 10, 15, 50, 100, 250]}]
    t0 = time.time()
    grid = grid_search_wrapper(SVC(),
                               X_train,
                               y_train,
                               params)
    tf = time.time()
    time_elapse_train = tf - t0

    # Test
    t0 = time.time()
    y_pred = grid.predict(X_test)
    tf = time.time()
    time_elapse_test = tf - t0
    test_acc = accuracy_score(y_test, y_pred)
    print(f'SVM test accuracy: {test_acc}')

    return [test_acc, time_elapse_train, time_elapse_test]


def knn_classifier(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series):
    # Train
    params = {'n_neighbors': [3, 5, 7, 9]}
    t0 = time.time()
    grid = grid_search_wrapper(KNeighborsClassifier(),
                               X_train,
                               y_train,
                               params)
    tf = time.time()
    time_elapse_train = tf - t0

    # Test
    t0 = time.time()
    y_pred = grid.predict(X_test)
    tf = time.time()
    time_elapse_test = tf - t0
    test_acc = accuracy_score(y_test, y_pred)
    print(f'KNN test accuracy: {test_acc}')

    return [test_acc, time_elapse_train, time_elapse_test]


def mlp_keras_classfier(X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        plot: bool = True):
    # Parameters
    VAL_SIZE = 0.15
    N_HIDDEN = 100
    LR = 0.0001
    EPOCHS = 100
    BATCH_SIZE = 16
    N_CLASSES = 6

    # Data
    import tensorflow as tf
    # Transform target labels with value between 0 and n_classes-1
    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    y_test_int = le.transform(y_test)
    # OneHotEncoder for classification in keras
    y_train_ohe = tf.keras.utils.to_categorical(y_train_int,
                                                num_classes=N_CLASSES)
    y_test_ohe = tf.keras.utils.to_categorical(y_test_int,
                                               num_classes=N_CLASSES)

    # Build the keras model
    model = Sequential([Input(X_train.shape[1]),
                        Dense(N_HIDDEN, activation='relu',
                              ),
                        Dense(N_CLASSES, activation='softmax')])

    # Train
    optimizer = Adam(lr=LR)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    t0 = time.time()
    history = model.fit(X_train.values,
                        y_train_ohe,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VAL_SIZE)
    tf = time.time()
    time_elapse_train = tf - t0

    # Plots
    if plot:
        plt.figure()
        plt.subplot(211)
        plt.ylabel('Cross Entropy Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.legend()

        plt.subplot(212)
        plt.ylabel('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='val')
        plt.legend()
        plt.tight_layout()

    # Test
    t0 = time.time()
    test_loss, test_acc = model.evaluate(X_test.values, y_test_ohe)
    tf = time.time()
    time_elapse_test = tf - t0
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    return [test_acc, time_elapse_train, time_elapse_test]


def main(data_path,
         n_runs: int = 10,
         test_size: float = 0.2,
         save: bool = True):
    # Data
    X, y, names = get_X_y_names(data_path)

    # Visualization
    t_sne = TSNE(n_components=2, learning_rate='auto',
                 init='random', perplexity=30, random_state=0)
    X_embedded = t_sne.fit_transform(X)
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(x=X_embedded[:, 0],
                    y=X_embedded[:, 1],
                    hue=y,
                    palette='tab10', ax=ax)
    ax.set_xlabel('tsne 1')
    ax.set_ylabel('tsne 2')

    test_acc_list = []
    for _ in tqdm(range(n_runs)):
        # Train and Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size)

        # Classifiers
        svm_clf = svm_classifiers(X_train, y_train, X_test, y_test)
        test_acc_list.append(['svm', svm_clf])
        knn_clf = knn_classifier(X_train, y_train, X_test, y_test)
        test_acc_list.append(['knn', knn_clf])
        mlp_clf = mlp_keras_classfier(X_train, y_train, X_test, y_test)
        test_acc_list.append(['mlp', mlp_clf])

    print(test_acc_list)
    df = pd.DataFrame(test_acc_list)
    df.columns = ['clf', 1]
    split_df = pd.DataFrame(df.iloc[:, 1].tolist(),
                            columns=['test_acc', 'train_time', 'test_time'])
    df = pd.concat([df.iloc[:, 0], split_df], axis=1)
    if save:
        df.to_csv(definitions.DATA_PROCESSED / 'test_data.csv')


if __name__ == '__main__':
    main(definitions.DATA_PROCESSED / 'final_dataset.csv')
