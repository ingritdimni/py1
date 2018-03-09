import numpy as np
import pandas as pd
from football_betting.my_utils import create_time_feature_from_season_and_stage, get_last_matches
from football_betting.fables import _create_simple_fable, _vectorize_simple_fable, simple_fable, \
    create_simple_relative_match_description, simple_stats_fable
from create_data import create_stationary_poisson_match_results


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
    print(match_results)
    # print(fables)

    fables2 = simple_stats_fable(match_results, nb_observed_match=2)
    print(fables2)
