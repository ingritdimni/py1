import numpy as np
import pandas as pd
from football_betting.create_data import create_stationary_poisson_match_results
from football_betting.my_utils import create_time_feature_from_season_and_stage, get_last_matches, split_inputs, \
    split_input, bkm_quote_to_probas


def test_bkm_to_probas():
    bkm_quotes = pd.DataFrame({'W': [3., 2.9, 2., 10.], 'D': [3., 2.9, 4., 9.], 'L': [3., 2.9, 4., 1.25]})
    probas = bkm_quote_to_probas(bkm_quotes, outcomes_labels=None)
    assert(probas[0, 0] == probas[0, 1] == probas[0, 2] == probas[1, 0] == probas[1, 1] == probas[1, 2])


def test_split():
    df_x = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'B': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'h']})
    df_y = df_x['B']

    x_80, x_20, (p80, p20) = split_input(df_x, 0.8, random=True, return_indices=True, seed=1)
    assert(np.array_equal(df_y[p20], pd.Series(['i', 'f'], name='B')))

    reduced_x, cross_val_x, reduced_y, cross_val_y = split_inputs(df_x, df_y, split_ratio=0.8, seed=1)
    assert(np.array_equal(df_y[p20], cross_val_y))


def test_get_last_matches():
    nb_teams = 8
    nb_seasons = 2
    match_results, actual_probas, team_params = create_stationary_poisson_match_results(nb_teams, nb_seasons, seed=0)

    # create and add time feature
    time_feature = create_time_feature_from_season_and_stage(match_results, base=100)
    match_results['date'] = time_feature
    l_time_feature = list(time_feature)
    assert (all(l_time_feature[i] <= l_time_feature[i + 1] for i in range(len(l_time_feature) - 1)))
    #print(match_results.tail(10))

    # search for last matches
    last_n_matches = get_last_matches(match_results, team=3, t=212, n=10, t_column_name='date',
                                      home_team_key='home_team_id', away_team_key='away_team_id')
    # print(last_n_matches.tail(10))
