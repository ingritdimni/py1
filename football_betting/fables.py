import numpy as np
import pandas as pd
from my_utils import get_last_matches


def simple_fable(match_results, nb_observed_match=20, padding=True, horizontal_features=False,
                 t_column_name='date', home_team_key='home_team_id', away_team_key='away_team_id',
                 home_goals_key='home_team_goal', away_goals_key='away_team_goal'):
    """ create simple fable and vectorize it """
    match_fables = match_results.apply(
        lambda x: _create_simple_fable(x, match_results, nb_observed_match=nb_observed_match,
                                       t_column_name=t_column_name, home_team_key=home_team_key,
                                       away_team_key=away_team_key, home_goals_key=home_goals_key,
                                       away_goals_key=away_goals_key), axis=1)
    # either creates horizontal features only, or an array of features (allow convolution)
    if horizontal_features:
        return _vectorize_simple_fable_horizontal(match_fables, nb_observed_match=nb_observed_match, padding=padding)
    else:
        fables = _vectorize_simple_fable(match_fables, nb_observed_match=nb_observed_match, padding=padding)
        fables = fables.reshape(list(fables.shape) + [1, ])
        return fables


def simple_stats_fable(match_results, nb_observed_match=20, t_column_name='date',
                       home_team_key='home_team_id', away_team_key='away_team_id',
                       home_goals_key='home_team_goal', away_goals_key='away_team_goal'):
    """ create simple stats fable and vectorize it """
    match_fables = match_results.apply(
        lambda x: _create_simple_fable(x, match_results, nb_observed_match=nb_observed_match,
                                       t_column_name=t_column_name, home_team_key=home_team_key,
                                       away_team_key=away_team_key, home_goals_key=home_goals_key,
                                       away_goals_key=away_goals_key), axis=1)
    return _vectorize_simple_stats_fable_horizontal(match_fables)


def _create_simple_fable(match, match_results, nb_observed_match=10, t_column_name='date',
                         home_team_key='home_team_id', away_team_key='away_team_id', home_goals_key='home_team_goal',
                         away_goals_key='away_team_goal'):
    """ returns chosen characteristics (=fable) for a given match.
     More precisely, call create_simple_relative_match_description which created a match description (dict) for each
     past match of the 2 involved teams.
     Doing so, a match history (list) is created by adding each match descr to the history of the home / away team.
     Then returns [home_team_match_history, away_team_match_history]"""

    home_team = match[home_team_key]
    away_team = match[away_team_key]
    t = match[t_column_name]
    home_last_matches = get_last_matches(match_results, home_team, t, nb_observed_match, t_column_name=t_column_name,
                                         home_team_key=home_team_key, away_team_key=away_team_key)
    away_last_matches = get_last_matches(match_results, away_team, t, nb_observed_match, t_column_name=t_column_name,
                                         home_team_key=home_team_key, away_team_key=away_team_key)
    home_matches_descr = home_last_matches.apply(lambda x: create_simple_relative_match_description(x, home_team,
                                                 home_team_key, away_team_key, home_goals_key, away_goals_key), axis=1)
    away_matches_descr = away_last_matches.apply(lambda x: create_simple_relative_match_description(x, away_team,
                                                 home_team_key, away_team_key, home_goals_key, away_goals_key), axis=1)

    # manage emptyness
    if not len(home_matches_descr):
        home_matches_descr = list()
    if not len(away_matches_descr):
        away_matches_descr = list()

    return [list(home_matches_descr), list(away_matches_descr)]


def _vectorize_simple_fable(match_fables, nb_observed_match=10, descr_size=2, padding=False):
    """ vectorization of simple fable (returns numpy array)"""
    padding_value = 1 if padding else np.nan
    vectorized_fables = np.full((match_fables.shape[0], nb_observed_match * 2, descr_size), padding_value)
    for i in range(len(match_fables)):
        home_team_fable = match_fables.iloc[i][0]
        away_team_fable = match_fables.iloc[i][1]

        for j, m_descr in enumerate(home_team_fable):
            vectorized_fables[i, j, 0] = m_descr['scored']
            vectorized_fables[i, j, 1] = m_descr['conceded']

        for j, m_descr in enumerate(away_team_fable):
            vectorized_fables[i, nb_observed_match+j, 0] = m_descr['scored']
            vectorized_fables[i, nb_observed_match+j, 1] = m_descr['conceded']

    return vectorized_fables


def _vectorize_simple_fable_horizontal(match_fables, nb_observed_match=10, descr_size=2, padding=False):
    """ vectorization of simple fable (returns numpy array)"""
    padding_value = 1 if padding else np.nan
    vectorized_fables = np.full((match_fables.shape[0], nb_observed_match * 2 * descr_size + 4), padding_value)
    for i in range(len(match_fables)):
        home_team_fable = match_fables.iloc[i][0]
        away_team_fable = match_fables.iloc[i][1]

        nb_wins, nb_defeats = 0, 0
        for j, m_descr in enumerate(home_team_fable):
            vectorized_fables[i, descr_size * j] = m_descr['scored']
            vectorized_fables[i, descr_size * j + 1] = m_descr['conceded']
            # on the below, count victories and defeats
            if m_descr['conceded'] < m_descr['scored']:
                nb_wins += 1
            elif m_descr['scored'] < m_descr['conceded']:
                nb_defeats += 1
        vectorized_fables[i, -4] = nb_wins
        vectorized_fables[i, -3] = nb_defeats

        nb_wins, nb_defeats = 0, 0
        for j, m_descr in enumerate(away_team_fable):
            vectorized_fables[i, nb_observed_match * 2 + descr_size * j] = m_descr['scored']
            vectorized_fables[i, nb_observed_match * 2 + descr_size * j + 1] = m_descr['conceded']
            # on the below, count victories and defeats
            if m_descr['conceded'] < m_descr['scored']:
                nb_wins += 1
            elif m_descr['scored'] < m_descr['conceded']:
                nb_defeats += 1
        vectorized_fables[i, -2] = nb_wins
        vectorized_fables[i, -1] = nb_defeats

    return vectorized_fables


def _vectorize_simple_stats_fable_horizontal(match_fables):
    s = np.zeros((match_fables.shape[0], 10))
    for i in range(len(match_fables)):
        s[i, 0], s[i, 1], s[i, 2], s[i, 3], s[i, 4] = relative_stats(match_fables.iloc[i][0])
        s[i, 5], s[i, 6], s[i, 7], s[i, 8], s[i, 9] = relative_stats(match_fables.iloc[i][1])
    return s


def create_simple_relative_match_description(match, reference_team, home_team_key='home_team_id',
                                             away_team_key='away_team_id', home_goals_key='home_team_goal',
                                             away_goals_key='away_team_goal'):
    home_team = match[home_team_key]
    away_team = match[away_team_key]
    assert(reference_team in (home_team, away_team))
    if home_team == reference_team:
        return {'scored': match[home_goals_key], 'conceded': match[away_goals_key]}
    return {'scored': match[away_goals_key], 'conceded': match[home_goals_key]}


def relative_stats(matches_descr, averaged=True):
    nb_wins, nb_draws, nb_defeats, sum_scored, sum_conceded = 0, 0, 0, 0, 0
    for j, m_descr in enumerate(matches_descr):
        sum_scored += m_descr['scored']
        sum_conceded += m_descr['conceded']
        if m_descr['conceded'] < m_descr['scored']:
            nb_wins += 1
        elif m_descr['scored'] < m_descr['conceded']:
            nb_defeats += 1
        else:
            nb_draws += 1
    if averaged:
        nb_obs = len(matches_descr)
        if nb_obs:
            nb_wins /= nb_obs
            nb_draws /= nb_obs
            nb_defeats /= nb_obs
            sum_scored /= nb_obs
            sum_conceded /= nb_obs
    return nb_wins, nb_draws, nb_defeats, sum_scored, sum_conceded

