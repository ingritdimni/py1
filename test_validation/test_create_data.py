import numpy as np
import pandas as pd

from football_betting.create_data import create_teams, update_teams_params, display_param_history, \
    create_stationary_poisson_match_results, create_dynamic_poisson_match_results, create_noisy_bookmaker_quotes


DATA_PATH = "D:/Football_betting/artificial_data/"  # default export path, might be overwritten
DISPLAY_GRAPH = False
actual_export = False
EPSILON = 10e-7


def test_param_dynamic():
    np.random.seed(0)
    nb_diffused_seasons = 3
    nb_steps_interseason = 10
    nb_teams = 18
    team_params = create_teams(nb_teams)
    cur_param = team_params
    #params_history = {t: [team_params[t], ] for t in range(nb_teams+1)[1:]}
    params_history = {t: list() for t in range(nb_teams + 1)[1:]}  # excludes first params
    for s in range(nb_diffused_seasons):
        for step in range(nb_steps_interseason):  # teams change at interseason --> more volatility ?!
            cur_param = update_teams_params(cur_param)
        for i in range((nb_teams-1) * 2):
            new_param = update_teams_params(cur_param)
            for t in range(nb_teams+1)[1:]:
                params_history[t].append(new_param[t])
            cur_param = new_param
    if DISPLAY_GRAPH:
        display_param_history(params_history, nb_diffused_seasons)


def test_export_stationary_poisson_match_results(path=DATA_PATH):
    """" exports stationary poisson match results and probabilities """
    nb_teams = 6
    nb_seasons = 5
    df_results, df_probas, actual_team_params = create_stationary_poisson_match_results(nb_teams, nb_seasons, seed=0)
    # print(df_results.tail(10))
    # print(df_probas.tail(10))
    if actual_export:
        df_results.to_csv(path + "stationary_poisson_results.csv", index=False)
        df_probas.to_csv(path + "stationary_poisson_results_probabilities.csv", index=False)


def test_export_dynamic_poisson_match_results(path=DATA_PATH):
    """" exports dynamic poisson match results and probabilities """
    nb_teams = 6
    nb_seasons = 5
    df_results, df_probas, params_history = create_dynamic_poisson_match_results(nb_teams, nb_seasons,
                                                                                 nb_fixed_seasons=1, seed=0)
    # print(df_results.tail(20))
    # print(df_probas.tail(20))
    if actual_export:
        df_results.to_csv(path + "dynamic_poisson_results.csv", index=False)
        df_probas.to_csv(path + "dynamic_poisson_results_probabilities.csv", index=False)
    if DISPLAY_GRAPH:
        display_param_history(params_history, nb_seasons)


def test_create_bookmaker_quotes():
    match_probas = pd.DataFrame({'W': [0.4, 0.6, 0.1, 1./3], 'D': [0.2, 0.2, 0.5, 1./3], 'L': [0.4, 0.2, 0.4, 1./3]})
    quotes = create_noisy_bookmaker_quotes(match_probas, fees=0.05, seed=0)
    sum_bkm_probs = quotes.apply(lambda x: np.reciprocal(x)).sum(axis=1)
    for i in range(sum_bkm_probs.shape[0]):
        assert(abs(sum_bkm_probs.iloc[0] - 1.05) < EPSILON)
    # print(quotes)
    # print(sum_bkm_probs)

