import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras import Model
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
import matplotlib.pyplot as plt
from all.create_data import create_stationary_poisson_match_results, create_dynamic_poisson_match_results
from my_utils import split_input, split_inputs
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score, log_loss
from functools import partial
import keras.backend as K
from itertools import product
import tensorflow as tf
import numpy as np
import warnings
warnings.simplefilter("ignore")

DATA_PATH = "D:/Football_betting/artificial_data/"
DISPLAY = True
DISPLAY_GRAPH = False
_EPSILON = 1e-7


def test_model_on_data():

    nb_teams = 18
    nb_seasons = 10

    np.random.seed(0)
    # match_results = pd.read_csv(DATA_PATH + "poisson_results.csv")
    # actual_probas = pd.read_csv(DATA_PATH + "poisson_results_match_probabilities.csv")
    # match_results, actual_probas, team_params = create_stationary_poisson_match_results(18, 10)
    match_results, actual_probas, team_params = create_dynamic_poisson_match_results(nb_teams, nb_seasons,
                                                                                     nb_fixed_seasons=1)



    # Split the train and the validation set for the fitting
    # X_train, X_val, Y_train, Y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=random_seed)
    match_results90, match_results10, (indices90, indices10) = split_input(match_results, split_ratio=0.9,
                                                                         random=False, return_indices=True)

    cur_season = 10
    cur_day = 1
    train_weights = exp_weights(match_results90, cur_season, cur_day, (nb_teams-1) * 2, season_rate=0.10)

    X_train, Y_train = trivial_feature_engineering(match_results90)
    X_val, Y_val = trivial_feature_engineering(match_results10)

    # x_data, y_data = trivial_feature_engineering(match_results)
    #
    # # Split the train and the validation set for the fitting
    # # X_train, X_val, Y_train, Y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=random_seed)
    # X_train, X_val, Y_train, Y_val, (indices90, indices10) = split_inputs(x_data, y_data, split_ratio=0.9,
    #                                                                       random=False, return_indices=True)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_val shape:", Y_val.shape)

    # get actual probabilities of issues for the validation set of matches
    actual_probas_train = actual_probas.iloc[indices90]
    actual_probas_val = actual_probas.iloc[indices10]
    print("best possible honnest score on train set:", log_loss(Y_train, actual_probas_train))
    print("best possible honnest score on validation set:", log_loss(Y_val, actual_probas_val))

    # define and configure model
    model = prepare_simple_nn_model(x_data.shape[1])

    # Its better to have a decreasing learning rate during the training to reach efficiently the global
    # minimum of the loss function.
    # To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically
    # every X steps (epochs) depending if it is necessary (when accuracy is not improved).
    # With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                patience=60,
                                                verbose=1,
                                                factor=0.6,
                                                min_lr=0.0001)

    # Define the optimizer
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    # Compile the model
    #model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")

    epochs = 250
    batch_size = 256
    history = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
                        verbose=2, callbacks=[learning_rate_reduction])

    # history = model.fit(x=[X_train, weights], y=Y_train, epochs=epochs, batch_size=batch_size,
    #                     validation_data=(X_val, Y_val), verbose=2, callbacks=[learning_rate_reduction])

    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'][5:], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'][5:], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    # ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    # ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    # legend = ax[1].legend(loc='best', shadow=True)
    if DISPLAY_GRAPH: plt.show()

    # model predictions
    predictions_val = model.predict(X_val)  # to get percentages
    predictions_train = model.predict(X_train)  # to get percentages
    #predictions = model.predict_on_batch(X_val)

    if DISPLAY:
        home_teams, away_teams = teams_from_dummies(X_val)
        print("--- on the below, few prediction examples")
        for i in range(min(X_val.shape[0], 10)):
            match_result = Y_val.iloc[i].idxmax(axis=1)
            #correct_prediction = actual_probas_val.iloc[i]
            print()
            print(home_teams.iloc[i], away_teams.iloc[i], '-->', match_result)
            print('predictions:', predictions_val[i])
            print('actual probs:', list(actual_probas_val.iloc[i]))
        print()
        print("best possible honnest score on train set:", round(log_loss(Y_train, actual_probas_train), 4))
        print("our score on training set               :", round(log_loss(Y_train, predictions_train), 4))
        print("best possible honnest score on validation set:", round(log_loss(Y_val, actual_probas_val), 4))
        print("our score on validation set                  :", round(log_loss(Y_val, predictions_val), 4))


def test_season_count_fraction():
    nb_teams = 20
    nb_match_per_season = (nb_teams - 1) * 2

    cur_season = 10
    cur_day = 17

    epsilon = 10e-7

    tested_inputs = [[9, 17, 1.0], [9, 13, 1.105263157894737], [9, 18, 0.9736842105263158],
                     [10, 2, 0.39473684210526316], [10, 17, 0.]]

    for s, d, expected_res in tested_inputs:
        print(s, d, season_count_fraction(s, d, cur_season, cur_day, nb_match_per_season))
        assert(abs(season_count_fraction(s, d, cur_season, cur_day, nb_match_per_season) - expected_res < epsilon))


def w_categorical_crossentropy(target, output, weights):
    # scale preds so that the class probas of each sample sum to 1 --> should not be necessary for us
    output /= tf.reduce_sum(output, len(output.get_shape()) - 1, True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(_EPSILON, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return - tf.reduce_sum(weights * target * tf.log(output), len(output.get_shape()) - 1)


def exp_weights(inputs, current_season, current_day, nb_days_per_season, season_rate=0.10,
                season_label='season', day_label='stage'):

    def local_exp_weight(s, d):  # shorter version of exp weights using context variables
        return exp_weight(season_rate, season_count_fraction(s, d, current_season, current_day, nb_days_per_season))

    weights = inputs.apply(lambda x: local_exp_weight(x[season_label], x[day_label]), axis=1)
    # weights = inputs.apply(lambda row: str(row[season_label]) + str(row[day_label]), axis=1)

    return weights


def exp_weight(season_rate, season_count_fraction):
    """ compute weight seen as an exponential discount factor, i.e. exp(-r * t), with r and t inputs"""
    return np.exp(- season_rate * season_count_fraction)


def season_count_fraction(past_season, past_day, current_season, current_day, nb_days_per_season):
    """ computes season count fraction between past season and days to current season and day """
    total_seasons = current_season - past_season
    total_days = current_day - past_day
    if total_days < 0:
        total_days += nb_days_per_season
        total_seasons -= 1
    assert(total_seasons >= 0)
    assert(total_days >= 0)
    return (total_seasons * nb_days_per_season + total_days) / nb_days_per_season


def teams_from_dummies(x_dummy, home_team_base_label="home_team_id", away_team_base_label="away_team_id"):
    """ convert match dummy vectorized description into two series: home_team_id and away_team_id """
    home_team_cols = [col for col in x_dummy.columns if home_team_base_label in col]
    away_team_cols = [col for col in x_dummy.columns if away_team_base_label in col]
    home_teams_id = x_dummy[home_team_cols].idxmax(axis=1).apply(lambda x: int(x[x.rfind('_') + 1:]))
    away_teams_id = x_dummy[away_team_cols].idxmax(axis=1).apply(lambda x: int(x[x.rfind('_') + 1:]))
    return home_teams_id, away_teams_id


def trivial_feature_engineering(full_data):
    """ basically, removes all features to just let:
        in inputs: home and away teams (vectorized as dummy vectors)
        in outputs: W D or L (Win Draw or Loss), seen as for the home team (vectorized as dummy vectors)"""
    labels_to_drop = ['season', 'stage', 'home_team_goal', 'away_team_goal']
    y_data = full_data.apply(get_match_label, axis=1)
    x_data = full_data.drop(labels=labels_to_drop, axis=1)
    x_dummy = pd.get_dummies(x_data, columns=['home_team_id', 'away_team_id'])
    y_dummy = pd.get_dummies(y_data, prefix_sep='')
    y_dummy = y_dummy[['W', 'D', 'L']]  # change order to get win first
    # print(x_dummy)
    # print(y_dummy)
    return x_dummy, y_dummy


def get_match_label(match):
    """ Derives a label for a given match. """

    # Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.DataFrame()
    #label.loc[0, 'match_api_id'] = match['match_api_id']

    # Identify match label
    if home_goals > away_goals:
        label.loc[0, ''] = "W"
    if home_goals == away_goals:
        label.loc[0, ''] = "D"
    if home_goals < away_goals:
        label.loc[0, ''] = "L"

    # Return label
    return label.loc[0]


def prepare_simple_nn_model(n_features):
    n_activations = 512
    base_dropout = 0.25
    l2_regularization_factor = 0.00005
    activation_fct = "sigmoid"
    model = Sequential()

    model.add(Dense(n_activations, activation=activation_fct, input_dim=n_features,
                    kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(base_dropout))

    model.add(Dense(n_activations, activation=activation_fct,
                    kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(base_dropout))

    # model.add(Dense(n_activations, activation=activation_fct,
    #                 kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    # model.add(Dropout(base_dropout))

    # model.add(Dense(n_activations, activation=activation_fct,
    #                 kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    # model.add(Dropout(base_dropout))

    model.add(Dense(n_activations, activation=activation_fct,
                    kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(base_dropout*1.3))
    model.add(Dense(3, activation="softmax"))
    return model


def prepare_weighted_nn_model(n_features, n_outputs):
    n_activations = 512
    base_dropout = 0.25
    l2_regularization_factor = 0.00005
    activation_fct = "sigmoid"

    input_layer = Input(shape=(n_features,))
    weights_tensor = Input(shape=(n_outputs,))

    x = Dense(n_activations, activation=activation_fct, input_dim=n_features,
              kernel_regularizer=regularizers.l2(l2_regularization_factor))(x)
    x = Dropout(base_dropout)(x)

    x = Dense(n_activations, activation=activation_fct, input_dim=n_features,
              kernel_regularizer=regularizers.l2(l2_regularization_factor))(x)
    x = Dropout(base_dropout)(x)

    x = Dense(n_activations, activation=activation_fct, input_dim=n_features,
              kernel_regularizer=regularizers.l2(l2_regularization_factor))(x)
    x = Dropout(base_dropout*1.3)(x)

    out = Dense(3, activation="softmax")(x)

    #w_custom_loss = partial(w_categorical_crossentropy, weights=weights_tensor)

    model = Model([input_layer, weights_tensor], out)
    return model

if __name__ == "__main__":
    #test_season_count_fraction()
    test_model_on_data()
