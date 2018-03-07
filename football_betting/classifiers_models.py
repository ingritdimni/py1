import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
from time import time

from create_data import full_data_creation
from my_utils import split_input


def test_train_classifier():
    nb_teams = 10
    nb_seasons = 10

    # load everything
    data = full_data_creation(nb_teams, nb_seasons, dynamic_tag="dynamic", nb_seasons_val=2, fable_observed_seasons=1,
                              bkm_noise=0.03, horizontal_fable_features=True)
    # split data
    X_train, X_val, Y_train, Y_val, actual_probas_train, actual_probas_val, bkm_quotes_train, bkm_quotes_val = data

    X_train, X_calib, [indices_train, indices_calib] = split_input(X_train, split_ratio=0.9, random=True,
                                                                   return_indices=True)
    Y_calib = Y_train.iloc[indices_calib]
    Y_train = Y_train.iloc[indices_train]

    dm_reduction = PCA()
    RF_clf = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')

    # Creating cross validation data splits
    cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
    cv_sets.get_n_splits(X_train, Y_train)

    n_features = X_train.shape[1]
    parameters_RF = {'clf__max_features': ['auto', 'log2'],
                     'dm_reduce__n_components': np.arange(5, n_features, np.around(n_features / 5))}

    # scorer = make_scorer(accuracy_score)
    scorer = make_scorer(log_loss)

    # computations core to use
    jobs = -1

    best_pipe = train_classifier(RF_clf, dm_reduction, X_train, Y_train, cv_sets, parameters_RF, scorer, jobs,
                                 use_grid_search=True, best_components=None, best_params=None)

    print(best_pipe)

    clf, dm_reduce, train_score, test_score = train_calibrate_predict(RF_clf, dm_reduction, X_train, Y_train, X_calib,
                                                                      Y_calib, X_val, Y_val, cv_sets, parameters_RF,
                                                                      scorer, jobs, use_grid_search=True)
    print(clf)
    print(dm_reduce)
    print(train_score)
    print(test_score)


def test():
    pass


def initialize_all_classifiers(n_features):
    # Initializing all models and parameters
    # Initializing classifiers
    RF_clf = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
    AB_clf = AdaBoostClassifier(n_estimators=200, random_state=2)
    GNB_clf = GaussianNB()
    KNN_clf = KNeighborsClassifier()
    LOG_clf = linear_model.LogisticRegression(multi_class="ovr", solver="sag", class_weight='balanced')
    classifiers = [RF_clf, AB_clf, GNB_clf, KNN_clf, LOG_clf]

    # Specficying scorer and parameters for grid search
    #scorer = make_scorer(accuracy_score)
    scorer = make_scorer(log_loss)
    parameters_RF = {'clf__max_features': ['auto', 'log2'],
                     'dm_reduce__n_components': np.arange(5, n_features, np.around(n_features / 5))}
    parameters_AB = {'clf__learning_rate': np.linspace(0.5, 2, 5),
                     'dm_reduce__n_components': np.arange(5, n_features, np.around(n_features / 5))}
    parameters_GNB = {'dm_reduce__n_components': np.arange(5, n_features, np.around(n_features / 5))}
    parameters_KNN = {'clf__n_neighbors': [3, 5, 10],
                      'dm_reduce__n_components': np.arange(5, n_features, np.around(n_features / 5))}
    parameters_LOG = {'clf__C': np.logspace(1, 1000, 5),
                      'dm_reduce__n_components': np.arange(5, n_features, np.around(n_features / 5))}

    parameters = {RF_clf: parameters_RF,
                  AB_clf: parameters_AB,
                  GNB_clf: parameters_GNB,
                  KNN_clf: parameters_KNN,
                  LOG_clf: parameters_LOG}

    # Initializing dimensionality reductions
    pca = PCA()
    dm_reductions = [pca]

    ## Training a baseline model and finding the best model composition using grid search
    # Train a simple GBC classifier as baseline model
    clf = LOG_clf
    clf.fit(X_train, y_train)
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                         accuracy_score(y_train, clf.predict(X_train))))
    print(
        "Score of {} for test set: {:.4f}.".format(clf.__class__.__name__, accuracy_score(y_test, clf.predict(X_test))))

    # Training all classifiers and comparing them
    clfs, dm_reductions, train_scores, test_scores = find_best_classifier(clfs, dm_reductions, scorer, X_train, y_train,
                                                                          X_calibrate, y_calibrate, X_test, y_test,
                                                                          cv_sets,
                                                                          parameters, n_jobs)

    # Plotting train and test scores
    plot_training_results(clfs, dm_reductions, np.array(train_scores), np.array(test_scores), path=path)


def find_best_classifier(classifiers, dm_reductions, scorer, X_t, y_t, X_c, y_c, X_v, y_v, cv_sets, params, jobs):
    """" Tune all classifier and dimensionality reduction combinations to find best classifier. """

    # Initialize result storage
    clfs_return = []
    dm_reduce_return = []
    train_scores = []
    test_scores = []

    # Loop through dimensionality reductions
    for dm in dm_reductions:

        # Loop through classifiers
        for clf in classifiers:
            # Grid search, calibrate, and test the classifier
            clf, dm_reduce, train_score, test_score = train_calibrate_predict(clf=clf, dm_reduction=dm, X_train=X_t,
                                                                              y_train=y_t,
                                                                              X_calibrate=X_c, y_calibrate=y_c,
                                                                              X_test=X_v, y_test=y_v, cv_sets=cv_sets,
                                                                              params=params[clf], scorer=scorer,
                                                                              jobs=jobs, use_grid_search=True)

            # Append the result to storage
            clfs_return.append(clf)
            dm_reduce_return.append(dm_reduce)
            train_scores.append(train_score)
            test_scores.append(test_score)

    # Return storage
    return clfs_return, dm_reduce_return, train_scores, test_scores


def train_calibrate_predict(clf, dm_reduction, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets,
                            params, scorer, jobs, use_grid_search=True, **kwargs):
    """ Train and predict using a classifer based on scorer. """

    # Indicate the classifier and the training set size
    print("Training a {} with {}...".format(clf.__class__.__name__, dm_reduction.__class__.__name__))

    # Train the classifier
    best_pipe = train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs,
                                 use_grid_search=use_grid_search)

    # Calibrate classifier
    print("Calibrating probabilities of classifier...")
    start = time()
    clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv='prefit', method='isotonic')
    clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    end = time()
    print("Calibrated {} in {:.1f}".format(clf.__class__.__name__, end - start))

    # Print the results of prediction for both training and testing
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                         predict_labels(clf, best_pipe, X_train, y_train)))
    print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__,
                                                     predict_labels(clf, best_pipe, X_test, y_test)))

    # Return classifier, dm reduction, and label predictions for train and test set
    return clf, best_pipe.named_steps['dm_reduce'], predict_labels(clf, best_pipe, X_train, y_train), predict_labels(
        clf, best_pipe, X_test, y_test)


def train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs, use_grid_search=True,
                     best_components=None, best_params=None):
    """ Fits a classifier to the training data. """

    # Start the clock, train the classifier, then stop the clock
    start = time()

    # Check if grid search should be applied
    if use_grid_search:
        # Define pipeline of dm reduction and classifier
        estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
        pipeline = Pipeline(estimators)

        # Grid search over pipeline and return best classifier
        # rpil: possible to define cv_sets here
        # cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
        # cv_sets.get_n_splits(X_train, y_train)
        params['dm_reduce__n_components'] = params['dm_reduce__n_components'].astype(int)
        grid_obj = model_selection.GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv_sets, n_jobs=jobs)
        grid_obj.fit(X_train, y_train)
        best_pipe = grid_obj.best_estimator_
    else:
        # Use best components that are known without grid search
        estimators = [('dm_reduce', dm_reduction(n_components=best_components)), ('clf', clf(best_params))]
        pipeline = Pipeline(estimators)
        best_pipe = pipeline.fit(X_train, y_train)

    end = time()

    # Print the results
    print("Trained {} in {:.1f} seconds".format(clf.__class__.__name__, end - start))

    # Return best pipe
    return best_pipe


# TODO: change criteria to crossentropy loss, not accuracy
def predict_labels(clf, best_pipe, features, target):
    """ Makes predictions using a fit classifier based on scorer. """

    # Start the clock, make predictions, then stop the clock
    y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))

    # Print and return results
    #return accuracy_score(target.values, y_pred)
    return log_loss(target.values, y_pred)  # rpil modif suggestion


if __name__ == "__main__":
    test_train_classifier()
