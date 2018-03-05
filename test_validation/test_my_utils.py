import numpy as np
import pandas as pd
from football_betting.create_data import create_stationary_poisson_match_results
from football_betting.my_utils import _create_simple_fable, create_time_feature_from_season_and_stage, \
    _vectorize_simple_fable, get_last_matches, simple_fable, split_inputs, split_input, \
    create_simple_relative_match_description


def test_create_full_fable():
    nb_teams = 10
    nb_seasons = 2
    match_results, actual_probas, team_params = create_stationary_poisson_match_results(nb_teams, nb_seasons, seed=0)

    # create and add time feature
    time_feature = create_time_feature_from_season_and_stage(match_results, base=100)
    match_results['date'] = time_feature

    # print(match_results.tail(10))
    nb_observed_match = 9
    match_fables = match_results.apply(
        lambda x: _create_simple_fable(x, match_results, nb_observed_match=nb_observed_match, t_column_name='date',
                                       home_team_key='home_team_id', away_team_key='away_team_id',
                                       home_goals_key='home_team_goal', away_goals_key='away_team_goal'), axis=1)

    # print(match_fables.tail(10))
    vect_fables = _vectorize_simple_fable(match_fables, nb_observed_match=nb_observed_match, padding=True)
    # print(vect_fables[-10:])


def test_debug_fable():

    # define keys vocabulary
    t_column = 'date'
    home_team_key, away_team_key = 'home_team_id', 'away_team_id'
    home_goals_key, away_goals_key = 'home_team_goal', 'away_team_goal'

    match_results = pd.DataFrame({t_column: [101, 102, 103], home_team_key: ['team_1', 'team_2', 'team_1'],
                                  away_team_key: ['team_4', 'team_4', 'team_4'],
                                  home_goals_key: [1, 2, 3], away_goals_key: [2, 3, 4]})
    # print(match_results)

    # print("\n### validate get_last_matches ###")
    last_n_matches = get_last_matches(match_results, team='team_4', t=103, n=10, t_column_name='date',
                                      home_team_key='home_team_id', away_team_key='away_team_id')
    # print(last_n_matches)

    # print("\n### validate create_simple_relative_match_description ###")
    away_team = 'team_4'
    away_last_matches = match_results
    past_matches_descr = away_last_matches.apply(lambda x: create_simple_relative_match_description(x, away_team,
                                                                               home_team_key, away_team_key,
                                                                               home_goals_key, away_goals_key), axis=1)
    # print(past_matches_descr)

    # print("\n### validate create_simple_fable (on given match)###")
    match_fable = _create_simple_fable(match_results.iloc[2], match_results, nb_observed_match=10, t_column_name='date',
                                       home_team_key='home_team_id', away_team_key='away_team_id',
                                       home_goals_key='home_team_goal', away_goals_key='away_team_goal')
    # print(match_fable)

    # print("\n### validate create_simple_fable (on several matches) ###")
    match_fables = match_results.apply(
        lambda x: _create_simple_fable(x, match_results, nb_observed_match=10, t_column_name='date',
                                       home_team_key='home_team_id', away_team_key='away_team_id',
                                       home_goals_key='home_team_goal', away_goals_key='away_team_goal'), axis=1)
    # print(match_fables)

    # print("\n### validate vectorize_simple_fable ###")
    vect_fables = _vectorize_simple_fable(match_fables, nb_observed_match=10, padding=True)
    # print(vect_fables[-10:])

    # print("\n### validate simple_fable ###")
    fables = simple_fable(match_results, nb_observed_match=2, padding=False)
    # print(match_results)
    # print(fables)


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
