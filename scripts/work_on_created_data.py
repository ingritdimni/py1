import numpy as np
import pandas as pd
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import matplotlib.pyplot as plt
from create_data import create_stationary_poisson_match_results, create_dynamic_poisson_match_results, \
    create_noisy_bookmaker_quotes
from my_utils import split_input, split_inputs, get_match_label, trivial_feature_engineering, simple_fable, \
    match_issues_hot_vectors, create_time_feature_from_season_and_stage
from sklearn.metrics import accuracy_score, log_loss
from nn_model import display_shapes, prepare_simple_nn_model, prepare_simple_nn_model_conv, \
    display_model_results_analysis
from invest_strategies import ConstantInvestStrategy, KellyInvestStrategy, ConstantStdDevInvestStrategy

import warnings
warnings.simplefilter("ignore")

DATA_PATH = "D:/Football_betting/artificial_data/"
MODEL_PATH = "D:/Football_betting/models/"
VERBOSE = True
DISPLAY_GRAPH = True
SAVE_MODEL = True
EPSILON = 1e-7


def end_to_end_test():
    nb_teams = 20
    nb_seasons = 20

    convolution_model = False

    #dynamic_tag = "stationary"
    dynamic_tag = "dynamic"
    params_str = 't' + str(nb_teams) + '_s' + str(nb_seasons) + '_'

    np.random.seed(0)
    try:
        match_results = pd.read_csv(DATA_PATH + params_str + dynamic_tag + "_poisson_results.csv")
        actual_probas = pd.read_csv(DATA_PATH + params_str + dynamic_tag + "_poisson_results_probabilities.csv")
        print(" ... data files have been loaded ...")
    except FileNotFoundError:
        print("no data files found: ... creating data ...")
        if dynamic_tag == "dynamic":
            match_results, actual_probas, team_params = create_dynamic_poisson_match_results(nb_teams, nb_seasons,
                                                                                             nb_fixed_seasons=2,
                                                                                             export=True)
        elif dynamic_tag == "stationary":
            match_results, actual_probas, team_params = create_stationary_poisson_match_results(nb_teams, nb_seasons,
                                                                                                export=True)

    bkm_quotes = create_noisy_bookmaker_quotes(actual_probas, std_dev=0.03, fees=0.05)

    match_results['date'] = create_time_feature_from_season_and_stage(match_results, base=100)
    print(" ... creating fables ...")
    match_fables = simple_fable(match_results, nb_observed_match=(nb_teams-1)*2)
    match_labels = match_issues_hot_vectors(match_results)

    # Split the train and the validation set for the fitting
    # X_train, X_val, Y_train, Y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=random_seed)
    X_train, X_val, (indices90, indices10) = split_input(match_fables, split_ratio=0.9,
                                                         random=False, return_indices=True)

    # eliminate first season (no fable)
    _, X_train, (_, remaining_train_indices) = split_input(X_train, split_ratio=1./9., random=False,
                                                           return_indices=True)

    Y_train = match_labels.iloc[indices90].iloc[remaining_train_indices]
    Y_val = match_labels.iloc[indices10]
    bkm_quotes_val = bkm_quotes.iloc[indices10]

    if VERBOSE: display_shapes(X_train, X_val, Y_train, Y_val)

    # get actual probabilities of issues for the validation set of matches
    actual_probas_train = actual_probas.iloc[indices90].iloc[remaining_train_indices]
    actual_probas_val = actual_probas.iloc[indices10]
    print("best possible honnest score on train set:", log_loss(Y_train, actual_probas_train))
    print("best possible honnest score on validation set:", log_loss(Y_val, actual_probas_val))

    # define and configure model
    if convolution_model:
        add_tag = "conv"
        n_activations = 64
        activation_fct = "sigmoid"
        dropout = 0.45
        l2_reg = 0.003
        model = prepare_simple_nn_model_conv(X_train.shape[1:], n_activations=n_activations, activation_fct=activation_fct,
                                        base_dropout=dropout, l2_regularization_factor=l2_reg)
    else:
        add_tag = "simple"
        n_activations = 128
        activation_fct = "sigmoid"
        dropout = 0.4
        l2_reg = 0.002
        model = prepare_simple_nn_model(X_train.shape[1:], n_activations=n_activations, activation_fct=activation_fct,
                                        base_dropout=dropout, l2_regularization_factor=l2_reg)

    # creates a model label containing most of its param (used to load / save it)
    model_label = add_tag + '_model_' + str(n_activations) + '_' + activation_fct + '_d' + str(dropout) + \
                  '_reg' + str(l2_reg) + '_shape_' + ''.join(str(e) + '_' for e in X_train.shape[1:] if e > 1)
    model_label = model_label[:-1].replace('.', '')
    model_label += '.h5py'

    # Its better to have a decreasing learning rate during the training to reach efficiently the global
    # minimum of the loss function.
    # To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically
    # every X steps (epochs) depending if it is necessary (when accuracy is not improved).
    # With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=60, verbose=1, factor=0.6, min_lr=0.0001)

    # Define the optimizer
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    epochs = 500
    batch_size = 256
    try:
        model = load_model(MODEL_PATH + model_label)
    except OSError:
        # Compile the model
        # model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer=optimizer, loss="categorical_crossentropy")

        if VERBOSE: model.summary()
        history = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
                            verbose=2, callbacks=[learning_rate_reduction])
        if SAVE_MODEL: model.save(MODEL_PATH + model_label)

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

    if VERBOSE:
        display_model_results_analysis(X_val, Y_val, predictions_val, actual_probas_val,
                                       [Y_train, predictions_train, actual_probas_train], nb_max_matchs_displayed=25)

    invest_strategy = ConstantInvestStrategy(1.)  # invest 1 in each match (if expected return > 1% actually)
    df_investments = invest_strategy.matches_investments(predictions_val, bkm_quotes_val)

    print(df_investments.tail(50))

    df_gains = invest_strategy.bet_gains(Y_val, df_investments)

    print(df_gains.tail(50))

if __name__ == "__main__":
    end_to_end_test()
