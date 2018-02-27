import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
import matplotlib.pyplot as plt
from all.create_data import create_minimalist_match_results, perfect_prediction
from my_utils import split_input, split_inputs
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
import warnings
warnings.simplefilter("ignore")

DATA_PATH = "D:/Football_betting/artificial_data/"
DISPLAY = True


def main():

    np.random.seed(0)
    # file_name = "poisson_results.csv"
    # full_data = pd.read_csv(DATA_PATH + file_name)

    full_data, actual_probas, team_params = create_minimalist_match_results(18, 10)

    x_data, y_data = trivial_feature_engineering(full_data)

    # Split the train and the validation set for the fitting
    # X_train, X_val, Y_train, Y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=random_seed)
    X_train, X_val, Y_train, Y_val, (indices90, indices10) = split_inputs(x_data, y_data, split_ratio=0.9,
                                                                          random=False, return_indices=True)
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_val shape:", Y_val.shape)

    # get actual probabilities of issues for the validation set of matches
    actual_probas_val = actual_probas.iloc[indices10]
    print("best possible honnest score on validation set:", log_loss(Y_val, actual_probas_val))

    # define and configure model
    model = prepare_model(x_data.shape[1])

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

    epochs = 200
    batch_size = 256
    history = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val),
                        verbose=2, callbacks=[learning_rate_reduction])

    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'][5:], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'][5:], color='r', label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    # ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    # ax[1].plot(history.history['val_acc'], color='r', label="Validation accuracy")
    # legend = ax[1].legend(loc='best', shadow=True)
    if DISPLAY: plt.show()

    # model predictions
    predictions_val = model.predict(X_val)  # to get percentages
    #predictions = model.predict_on_batch(X_val)

    print(predictions_val)
    home_teams, away_teams = teams_from_dummies(X_val)
    # best_predictions = X_val.apply(lambda x:perfect_prediction(teams_from_dummies(x), team_params))
    # print(best_predictions)

    # for i in range(min(X_val.shape[0], 20)):
    #     match_result = Y_val.iloc[i].idxmax(axis=1)
    #     home_team_id, away_team_id = home_teams.iloc[i], away_teams.iloc[i]
    #     #correct_prediction = perfect_prediction(home_team_id, away_team_id, team_params)
    #     correct_prediction = actual_probas_val.iloc[i]
    #     print()
    #     print(home_team_id, away_team_id, match_result)
    #     print('predictions:', predictions[i])
    #     print('actual probs:', correct_prediction)


def teams_from_dummies(x_dummy, home_team_base_label="home_team_id", away_team_base_label="away_team_id"):
    home_team_cols = [col for col in x_dummy.columns if home_team_base_label in col]
    away_team_cols = [col for col in x_dummy.columns if away_team_base_label in col]
    home_teams_id = x_dummy[home_team_cols].idxmax(axis=1).apply(lambda x: int(x[x.rfind('_') + 1:]))
    away_teams_id = x_dummy[away_team_cols].idxmax(axis=1).apply(lambda x: int(x[x.rfind('_') + 1:]))
    return home_teams_id, away_teams_id


def trivial_feature_engineering(full_data):
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
    ''' Derives a label for a given match. '''

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


def prepare_model(n_features):
    n_activations = 512
    base_dropout = 0.2
    l2_regularization_factor = 0.005*0.
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

if __name__ == "__main__":
    main()
