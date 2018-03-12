import numpy as np
import pandas as pd
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model
import matplotlib.pyplot as plt
from my_utils import split_input, split_inputs, get_match_label, trivial_feature_engineering, \
    match_issues_hot_vectors, create_time_feature_from_season_and_stage, contain_nan
from sklearn.metrics import accuracy_score, log_loss
from fables import simple_fable, simple_stats_fable
from nn_model import display_shapes, prepare_simple_nn_model, prepare_simple_nn_model_conv, \
    display_model_results_analysis, display_results_analysis
from invest_strategies import ConstantAmountInvestStrategy, KellyInvestStrategy, ConstantStdDevInvestStrategy, \
    ConstantPercentInvestStrategy
from data_preparation import simple_data_prep


MODEL_PATH = "D:/Football_betting/models/"
VERBOSE = True
DISPLAY_GRAPH = True
SAVE_MODEL = True
EPSILON = 1e-7


def end_to_end_test():
    np.random.seed(0)

    # load data
    match_data, match_features, match_labels, bkm_quotes = simple_data_prep(verbose=1, fable_observed_matches=40,
                                                                            padding=False, fable="match_hist",
                                                                            label_format="hot_vectors")

    # split data
    split_ratio = 0.15
    X_train, X_val, (indices_train, indices_val) = split_input(match_features, 1.-split_ratio, random=True,
                                                               return_indices=True)
    Y_train, Y_val = match_labels.iloc[indices_train], match_labels.iloc[indices_val]
    bkm_quotes_train, bkm_quotes_val = bkm_quotes.iloc[indices_train], bkm_quotes.iloc[indices_val]

    display_shapes(X_train, X_val, Y_train, Y_val)

    epochs = 200
    convolution_model = False
    # define and configure model
    if convolution_model:
        add_tag = "conv"
        # n_activations = 64
        n_activations = 16
        n_conv_filter = 4
        # activation_fct = "sigmoid"
        activation_fct = "relu"
        dropout = 0.45
        # l2_reg = 0.003
        l2_reg = 0.07
        model = prepare_simple_nn_model_conv(X_train.shape[1:], n_activations=n_activations, n_conv_filter=n_conv_filter,
                                             activation_fct=activation_fct, base_dropout=dropout,
                                             l2_regularization_factor=l2_reg)
    else:
        add_tag = "simple"
        # n_activations = 128  # sigmoid
        n_activations = 50  # relu
        # activation_fct = "sigmoid"
        activation_fct = "relu"
        dropout = 0.45
        # l2_reg = 0.002  # sigmoid
        l2_reg = 0.05  # relu
        model = prepare_simple_nn_model(X_train.shape[1:], n_activations=n_activations, activation_fct=activation_fct,
                                        base_dropout=dropout, l2_regularization_factor=l2_reg)

    # creates a model label containing most of its param (used to load / save it)
    model_label = add_tag + '_model_' + str(n_activations) + '_' + activation_fct + '_d' + str(dropout) + '_reg' + \
                  str(l2_reg) + '_shape_' + ''.join(str(e) + '_' for e in X_train.shape[1:] if e > 1) + 'e' + \
                  str(epochs)
    model_label = model_label.replace('.', '')
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
        print("\n --- TRAIN ANALYSIS --- ")
        display_results_analysis(Y_train, predictions_train, bkm_quotes_train, nb_max_matchs_displayed=0)
        print("\n --- VAL ANALYSIS --- ")
        display_results_analysis(Y_val, predictions_val, bkm_quotes_val, nb_max_matchs_displayed=0)

    # on the below, reduce universe to matches with quotes
    remove_nan_mask_val = [not contain_nan(bkm_quotes_val.iloc[i]) for i in range(bkm_quotes_val.shape[0])]
    bkm_quotes_val_r = bkm_quotes_val.iloc[remove_nan_mask_val]
    Y_val_r = Y_val.iloc[remove_nan_mask_val]
    predictions_val_r = predictions_val[remove_nan_mask_val]

    constant_invest_stgy = ConstantAmountInvestStrategy(1.)  # invest 1 in each match (if expected return > 1% actually)
    constant_sigma_invest_stgy = ConstantStdDevInvestStrategy(0.01)  # stdDev of each bet is 1% of wealth
    kelly_invest_stgy = KellyInvestStrategy()  # Kelly's ratio investment to maximize's wealth long term return
    constant_percent_stgy = ConstantPercentInvestStrategy(0.01)  # invest 1% of money each time

    for invest_stgy in [constant_invest_stgy, constant_sigma_invest_stgy, kelly_invest_stgy, constant_percent_stgy]:
        print("\n#### results for ", invest_stgy.__class__.__name__, "####")
        init_wealth = 100
        df_recap_stgy = invest_stgy.apply_invest_strategy(predictions_val_r, bkm_quotes_val_r, Y_val_r,
                                                          init_wealth=init_wealth)

        print(df_recap_stgy[['invested_amounts', 'exp_gain_amounts', 'gain_amounts']].sum())
        print('wealth: from', init_wealth, 'to', round(df_recap_stgy['wealth'].iloc[-1], 4))

if __name__ == "__main__":
    end_to_end_test()