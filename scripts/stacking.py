import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Conv1D
from keras import regularizers
from sklearn.metrics import log_loss
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

from stathelper import PoissonHelper
from my_utils import match_outcomes_hot_vectors, split_input, bkm_quote_to_probas
from utils_generic import contain_nan


PATH = "D:/Football_betting/tmp/"
DISPLAY_GRAPH = True


def stacking_predictions():

    np.random.seed(2)

    nn_pred = np.genfromtxt("D:/Football_betting/predictions/" + 'conv_nn_predictions.csv', delimiter=',')
    dixon_pred = np.genfromtxt("D:/Football_betting/predictions/" + 'dixon_coles_predictions.csv', delimiter=',')
    bkm_quotes = pd.read_csv("D:/Football_betting/predictions/" + 'bookmaker_quotes.csv', header=0)
    result_labels = pd.read_csv("D:/Football_betting/predictions/" + 'actual_results.csv', header=0)
    bkm_probas = bkm_quote_to_probas(bkm_quotes)

    # on the below, reduce universe to matches with quotes
    remove_nan_mask_val = [not contain_nan(bkm_probas[i]) for i in range(bkm_probas.shape[0])]
    bkm_probas = bkm_probas[remove_nan_mask_val]
    nn_pred = nn_pred[remove_nan_mask_val]
    dixon_pred = dixon_pred[remove_nan_mask_val]
    result_labels = result_labels.iloc[remove_nan_mask_val]

    y_hot_vectors_train, y_hot_vectors_val, (indices_train, indices_val) = split_input(result_labels, split_ratio=0.8,
                                                                                       random=True, return_indices=True)

    bkm_probas_train, bkm_probas_val = bkm_probas[indices_train], bkm_probas[indices_val]
    nn_pred_train, nn_pred_val = nn_pred[indices_train], nn_pred[indices_val]
    dixon_pred_train, dixon_pred_val = dixon_pred[indices_train], dixon_pred[indices_val]

    x_train = np.concatenate(tuple([bkm_probas_train, nn_pred_train, dixon_pred_train]), axis=1)
    x_val = np.concatenate(tuple([bkm_probas_val, nn_pred_val, dixon_pred_val]), axis=1)
    y_train = y_hot_vectors_train
    y_val = y_hot_vectors_val

    print('inputs shapes')
    print('x_train', x_train.shape)
    print('x_val', x_val.shape)
    print('y_train', y_train.shape)
    print('y_val', y_val.shape)

    n_activations = 20
    model = simple_stacking_nn_model(n_activations, x_train.shape[1:], l2_regularization_factor=0.00005,
                                     dropout_factor=0.3)

    # Define the optimizer
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy")

    # Its better to have a decreasing learning rate during the training to reach efficiently the global
    # minimum of the loss function.
    # To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically
    # every X steps (epochs) depending if it is necessary (when accuracy is not improved).
    # With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=60, verbose=1, factor=0.6, min_lr=0.0001)

    epochs = 800
    batch_size = 512  # all ?

    # if VERBOSE: model.summary()
    history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        verbose=2, callbacks=[learning_rate_reduction])

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
    predictions_val = model.predict(x_val)  # to get percentages
    predictions_train = model.predict(x_train)  # to get percentages

    print('predictions_train  ; ', log_loss(y_train, predictions_train))
    print('predictions_val    ; ', log_loss(y_val, predictions_val))
    print('bkm_train    ; ', log_loss(y_train, bkm_probas_train))
    print('bkm_val      ; ', log_loss(y_val, bkm_probas_val))
    print('--')
    print('nn_train    ; ', log_loss(y_train, nn_pred_train))
    print('nn_val      ; ', log_loss(y_val, nn_pred_val))
    print('dx_train    ; ', log_loss(y_train, dixon_pred_train))
    print('dx_val      ; ', log_loss(y_val, dixon_pred_val))


def test_stacking_model():

    # load data
    y_hot_vectors = pd.read_csv(PATH + "labels.csv")
    perfect_preds = pd.read_csv(PATH + "perfect_pred.csv")
    noisy_preds = pd.read_csv(PATH + "noisy_pred.csv")
    wrong_preds = pd.read_csv(PATH + "wrong_pred.csv")

    print('perfect; ', log_loss(y_hot_vectors, perfect_preds))
    print('noisy  ; ', log_loss(y_hot_vectors, noisy_preds))
    print('wrong  ; ', log_loss(y_hot_vectors, wrong_preds))

    np.random.seed(2)

    perfect_preds_train, perfect_preds_val, (indices_train, indices_val)= split_input(perfect_preds, split_ratio=0.8,
                                                                                      random=True, return_indices=True)
    y_hot_vectors_train, y_hot_vectors_val = y_hot_vectors.iloc[indices_train], y_hot_vectors.iloc[indices_val]
    noisy_preds_train, noisy_preds_val = noisy_preds.iloc[indices_train], noisy_preds.iloc[indices_val]
    wrong_preds_train, wrong_preds_val = wrong_preds.iloc[indices_train], wrong_preds.iloc[indices_val]

    x_train = np.concatenate(tuple([noisy_preds_train, wrong_preds_train]), axis=1)
    x_val = np.concatenate(tuple([noisy_preds_val, wrong_preds_val]), axis=1)
    y_train = y_hot_vectors_train
    y_val = y_hot_vectors_val

    n_activations = 20
    model = simple_stacking_nn_model(n_activations, x_train.shape[1:], l2_regularization_factor=0.00005,
                                     dropout_factor=0.3)

    # Define the optimizer
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy")

    # Its better to have a decreasing learning rate during the training to reach efficiently the global
    # minimum of the loss function.
    # To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically
    # every X steps (epochs) depending if it is necessary (when accuracy is not improved).
    # With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=60, verbose=1, factor=0.6, min_lr=0.0001)

    epochs = 800
    batch_size = 512  # all ?

    # if VERBOSE: model.summary()
    history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        verbose=2, callbacks=[learning_rate_reduction])

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
    predictions_val = model.predict(x_val)  # to get percentages
    predictions_train = model.predict(x_train)  # to get percentages

    print('perfect; ', log_loss(y_hot_vectors, perfect_preds))
    print('noisy  ; ', log_loss(y_hot_vectors, noisy_preds))
    print('wrong  ; ', log_loss(y_hot_vectors, wrong_preds))

    print('predictions_train  ; ', log_loss(y_train, predictions_train))
    print('predictions_val    ; ', log_loss(y_val, predictions_val))
    print('perfect_val    ; ', log_loss(y_val, perfect_preds_val))


def create_data(export=True):

    n = 10000
    min_pram = 0.5
    max_param = 1.8
    np.random.seed(1)
    params_home = np.clip(np.random.randn(n, 1)*0.3 + 1.2, min_pram, max_param)
    params_away = np.clip(np.random.randn(n, 1) * 0.3 + 1, min_pram, max_param)
    # print(params_home)
    # home_goals, away_goals = PoissonHelper.play_match(home_param, away_param)

    match_results = pd.DataFrame(columns=['home_team_goal', 'away_team_goal'])
    perfect_preds = pd.DataFrame(columns=['W', 'D', 'L'])
    noisy_preds = pd.DataFrame(columns=['W', 'D', 'L'])
    wrong_preds = pd.DataFrame(columns=['W', 'D', 'L'])
    for i in range(n):
        home_goals, away_goals = PoissonHelper.play_match(params_home[i], params_away[i])
        p_win, p_draw, p_loss = [float(e) for e in
                                 PoissonHelper.match_outcomes_probabilities(params_home[i], params_away[i])]
        match_results = match_results.append({'home_team_goal': home_goals, 'away_team_goal': away_goals},
                                             ignore_index=True)
        perfect_preds = perfect_preds.append({'W': p_win, 'D': p_draw, 'L': p_loss}, ignore_index=True)

        # noisy neutral oriented preds
        noise = 0.02
        p_noisy_win = float(max(min(p_win + noise * np.random.randn(), 1.), 0))
        p_noisy_loss = float(max(min(p_loss + noise * np.random.randn(), 1.), 0))
        p_noisy_draw = 1. - p_noisy_win - p_noisy_loss
        noisy_preds = noisy_preds.append({'W': p_noisy_win, 'D': p_noisy_draw, 'L': p_noisy_loss}, ignore_index=True)

        # noisy bad oriented preds
        home_win_sign = (home_goals - away_goals) / abs(home_goals - away_goals) if home_goals - away_goals != 0 else 0
        noise = 0.03
        wrong_direction = 0.01
        p_noisy_win_wrong = float(max(min(p_win + noise * np.random.randn() - wrong_direction * home_win_sign, 1.), 0))
        p_noisy_loss_wrong = float(max(min(p_loss + noise * np.random.randn() + wrong_direction * home_win_sign, 1.), 0))
        p_noisy_draw_wrong = 1. - p_noisy_win_wrong - p_noisy_loss_wrong
        wrong_preds = wrong_preds.append({'W': p_noisy_win_wrong, 'D': p_noisy_draw_wrong, 'L': p_noisy_loss_wrong},
                                         ignore_index=True)

    y_hot_vectors = match_outcomes_hot_vectors(match_results)

    print('perfect; ', log_loss(y_hot_vectors, perfect_preds))
    print('noisy  ; ', log_loss(y_hot_vectors, noisy_preds))
    print('wrong  ; ', log_loss(y_hot_vectors, wrong_preds))

    if export:
        y_hot_vectors.to_csv(PATH + "labels.csv", index=False)
        perfect_preds.to_csv(PATH + "perfect_pred.csv", index=False)
        noisy_preds.to_csv(PATH + "noisy_pred.csv", index=False)
        wrong_preds.to_csv(PATH + "wrong_pred.csv", index=False)


def simple_stacking_nn_model(n_activations, input_shape, activation_fct="relu", l2_regularization_factor=0.,
                             dropout_factor=0.):
    model = Sequential()

    if len(input_shape) > 1:
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(n_activations, activation=activation_fct,
                        kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    else:
        model.add(Dense(n_activations, activation=activation_fct, input_shape=input_shape,
                        kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(dropout_factor))

    model.add(Dense(n_activations, activation=activation_fct,
                    kernel_regularizer=regularizers.l2(l2_regularization_factor)))
    model.add(Dropout(dropout_factor))
    model.add(Dense(3, activation="softmax"))
    return model

if __name__ == "__main__":
    # create_data()
    # test_stacking_model()
    stacking_predictions()