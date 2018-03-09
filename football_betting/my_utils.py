import numpy as np
import pandas as pd


def trivial_feature_engineering(match_results, home_team_key='home_team_id', away_team_key='away_team_id'):
    """ basically, removes all features to just let:
        in inputs: home and away teams (vectorized as dummy vectors)
        in outputs: W D or L (Win Draw or Loss), seen as for the home team (vectorized as dummy vectors)"""

    labels_to_drop = list(set(match_results.columns.values) - {home_team_key, away_team_key})
    x_data = match_results.drop(labels=labels_to_drop, axis=1)
    x_dummy = pd.get_dummies(x_data, columns=[home_team_key, away_team_key])

    y_dummy = match_issues_hot_vectors(match_results)
    return x_dummy, y_dummy


def match_issues_hot_vectors(match_results, home_goals_key='home_team_goal', away_goal_key='away_team_goal'):
    y_data = match_results.apply(lambda x: get_match_label(x, home_goals_key, away_goal_key), axis=1)
    y_dummy = pd.get_dummies(y_data, prefix_sep='')
    y_dummy = y_dummy[['W', 'D', 'L']]  # change order to get win first
    return y_dummy


def match_issues_indices(match_results, home_goals_key='home_team_goal', away_goal_key='away_team_goal'):
    y_indices = match_results.apply(lambda x: get_index_match_label(x, home_goals_key, away_goal_key), axis=1)
    return y_indices


def display_shapes(X_train, X_val, Y_train, Y_val):
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_val shape:", Y_val.shape)


def get_match_label(match, home_goals_key='home_team_goal', away_goal_key='away_team_goal'):
    """ Derives a label for a given match. """
    home_goals = match[home_goals_key]
    away_goals = match[away_goal_key]
    if home_goals > away_goals:
        return "W"
    if home_goals == away_goals:
        return "D"
    else:
        return "L"


def get_index_match_label(match, home_goals_key='home_team_goal', away_goal_key='away_team_goal'):
    """ Derives a label for a given match. """
    home_goals = match[home_goals_key]
    away_goals = match[away_goal_key]
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    else:
        return 2

#
# def simple_fable(match_results, nb_observed_match=20, padding=True, horizontal_features=False,
#                  t_column_name='date', home_team_key='home_team_id', away_team_key='away_team_id',
#                  home_goals_key='home_team_goal', away_goals_key='away_team_goal'):
#     """ create simple fable and vectorize it """
#     match_fables = match_results.apply(
#         lambda x: _create_simple_fable(x, match_results, nb_observed_match=nb_observed_match,
#                                        t_column_name=t_column_name, home_team_key=home_team_key,
#                                        away_team_key=away_team_key, home_goals_key=home_goals_key,
#                                        away_goals_key=away_goals_key), axis=1)
#     # either creates horizontal features only, or an array of features (allow convolution)
#     if horizontal_features:
#         return _vectorize_simple_fable_horizontal(match_fables, nb_observed_match=nb_observed_match, padding=padding)
#     else:
#         fables = _vectorize_simple_fable(match_fables, nb_observed_match=nb_observed_match, padding=padding)
#         fables = fables.reshape(list(fables.shape) + [1, ])
#         return fables


# def _create_simple_fable(match, match_results, nb_observed_match=10, t_column_name='date',
#                          home_team_key='home_team_id', away_team_key='away_team_id', home_goals_key='home_team_goal',
#                          away_goals_key='away_team_goal'):
#     """ returns chosen characteristics (=fable) for a given match.
#      More precisely, call create_simple_relative_match_description which created a match description (dict) for each
#      past match of the 2 involved teams.
#      Doing so, a match history (list) is created by adding each match descr to the history of the home / away team.
#      Then returns [home_team_match_history, away_team_match_history]"""
#
#     home_team = match[home_team_key]
#     away_team = match[away_team_key]
#     t = match[t_column_name]
#     home_last_matches = get_last_matches(match_results, home_team, t, nb_observed_match, t_column_name=t_column_name,
#                                          home_team_key=home_team_key, away_team_key=away_team_key)
#     away_last_matches = get_last_matches(match_results, away_team, t, nb_observed_match, t_column_name=t_column_name,
#                                          home_team_key=home_team_key, away_team_key=away_team_key)
#     home_matches_descr = home_last_matches.apply(lambda x: create_simple_relative_match_description(x, home_team,
#                                                  home_team_key, away_team_key, home_goals_key, away_goals_key), axis=1)
#     away_matches_descr = away_last_matches.apply(lambda x: create_simple_relative_match_description(x, away_team,
#                                                  home_team_key, away_team_key, home_goals_key, away_goals_key), axis=1)
#
#     # manage emptyness
#     if not len(home_matches_descr):
#         home_matches_descr = list()
#     if not len(away_matches_descr):
#         away_matches_descr = list()
#
#     return [list(home_matches_descr), list(away_matches_descr)]
#
#
# def _vectorize_simple_fable(match_fables, nb_observed_match=10, descr_size=2, padding=False):
#     """ vectorization of simple fable (returns numpy array)"""
#     padding_value = 1 if padding else np.nan
#     vectorized_fables = np.full((match_fables.shape[0], nb_observed_match * 2, descr_size), padding_value)
#     for i in range(len(match_fables)):
#         home_team_fable = match_fables.iloc[i][0]
#         away_team_fable = match_fables.iloc[i][1]
#
#         for j, m_descr in enumerate(home_team_fable):
#             vectorized_fables[i, j, 0] = m_descr['scored']
#             vectorized_fables[i, j, 1] = m_descr['conceded']
#
#         for j, m_descr in enumerate(away_team_fable):
#             vectorized_fables[i, nb_observed_match+j, 0] = m_descr['scored']
#             vectorized_fables[i, nb_observed_match+j, 1] = m_descr['conceded']
#
#     return vectorized_fables
#
#
# def _vectorize_simple_fable_horizontal(match_fables, nb_observed_match=10, descr_size=2, padding=False):
#     """ vectorization of simple fable (returns numpy array)"""
#     padding_value = 1 if padding else np.nan
#     vectorized_fables = np.full((match_fables.shape[0], nb_observed_match * 2 * descr_size + 4), padding_value)
#     for i in range(len(match_fables)):
#         home_team_fable = match_fables.iloc[i][0]
#         away_team_fable = match_fables.iloc[i][1]
#
#         nb_wins, nb_defeats = 0, 0
#         for j, m_descr in enumerate(home_team_fable):
#             vectorized_fables[i, descr_size * j] = m_descr['scored']
#             vectorized_fables[i, descr_size * j + 1] = m_descr['conceded']
#             # on the below, count victories and defeats
#             if m_descr['conceded'] < m_descr['scored']:
#                 nb_wins += 1
#             elif m_descr['scored'] < m_descr['conceded']:
#                 nb_defeats += 1
#         vectorized_fables[i, -4] = nb_wins
#         vectorized_fables[i, -3] = nb_defeats
#
#         nb_wins, nb_defeats = 0, 0
#         for j, m_descr in enumerate(away_team_fable):
#             vectorized_fables[i, nb_observed_match * 2 + descr_size * j] = m_descr['scored']
#             vectorized_fables[i, nb_observed_match * 2 + descr_size * j + 1] = m_descr['conceded']
#             # on the below, count victories and defeats
#             if m_descr['conceded'] < m_descr['scored']:
#                 nb_wins += 1
#             elif m_descr['scored'] < m_descr['conceded']:
#                 nb_defeats += 1
#         vectorized_fables[i, -2] = nb_wins
#         vectorized_fables[i, -1] = nb_defeats
#
#     return vectorized_fables
#

# def create_simple_relative_match_description(match, reference_team, home_team_key='home_team_id',
#                                              away_team_key='away_team_id', home_goals_key='home_team_goal',
#                                              away_goals_key='away_team_goal'):
#     home_team = match[home_team_key]
#     away_team = match[away_team_key]
#     assert(reference_team in (home_team, away_team))
#     if home_team == reference_team:
#         return {'scored': match[home_goals_key], 'conceded': match[away_goals_key]}
#     return {'scored': match[away_goals_key], 'conceded': match[home_goals_key]}


def create_time_feature_from_season_and_stage(match_results, base=100, season_key='season', stage_key='stage'):
    """ create a fake date feature from season and stage. It will actually be an int, but order will be correct"""
    def create_date_from_season_and_stage(season, stage, base):
        assert(stage < base)
        return season * base + stage
    match_time = match_results.apply(lambda x: create_date_from_season_and_stage(x[season_key], x[stage_key], base),
                                     axis=1)
    return match_time


def get_last_matches(match_results, team, t, n, t_column_name='date', home_team_key='home_team_id',
                     away_team_key='away_team_id'):
    """ Get the last n matches of a given team """

    # Filter team matches from matches
    team_matches = match_results[(match_results[home_team_key] == team) | (match_results[away_team_key] == team)]

    # Filter x last matches from team matches
    last_n_matches = team_matches[team_matches[t_column_name] < t].sort_values(by=t_column_name,
                                                                               ascending=False).iloc[0:n, :]

    return last_n_matches


def split_input(x, split_ratio, random=True, seed=None, return_indices=False):
    """ return reduced_x, cross_val_x and optionally involved_indices
    reduced_x has size split_ratio * x.shape[0]
    cross_val_x has size (1-split_ratio) * x.shape[0]
    involved_indices (optional) is a list of 2 elements: first and second set of involved_indices"""
    if seed: np.random.seed(seed)
    assert(0. <= split_ratio <= 1.)

    if random:
        # shuffle inputs
        p = np.random.permutation(x.shape[0])
        if isinstance(x, np.ndarray):
            x = x[p]
        elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.iloc[p]
        else:
            raise TypeError("input type should be ndarray or DataFrame or Series")
    else:
        p = np.array(range(x.shape[0]))

    # split our training all into 2 sets
    n = int(x.shape[0] * split_ratio)
    cross_val_x = x[n:]
    reduced_x = x[:n]

    if return_indices:
        return reduced_x, cross_val_x, [p[:n], p[n:]]
    return reduced_x, cross_val_x


def split_inputs(x, y, split_ratio, random=True, seed=None, return_indices=False):
    """ same as split_input, with two inputs x and y to split"""
    if seed: np.random.seed(seed)
    assert(x.shape[0] == y.shape[0])

    if random:
        # shuffle inputs
        p = np.random.permutation(x.shape[0])
        if isinstance(x, np.ndarray):
            x = x[p]
            y = y[p]
        elif isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.iloc[p]
            y = y.iloc[p]
        else:
            raise TypeError("input type should be ndarray or DataFrame or Series")
    else:
        p = np.array(range(x.shape[0]))

    # split our training all into 2 sets
    n = int(x.shape[0] * split_ratio)
    cross_val_x = x[n:]
    reduced_x = x[:n]
    cross_val_y = y[n:]
    reduced_y = y[:n]

    if return_indices:
        return reduced_x, cross_val_x, reduced_y, cross_val_y, [p[:n], p[n:]]
    return reduced_x, cross_val_x, reduced_y, cross_val_y


if __name__ == "__main__":
    # test_split()
    # test_get_last_matches()
    test_debug_fable()
    # test_create_full_fable()
