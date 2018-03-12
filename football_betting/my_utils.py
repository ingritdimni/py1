import numpy as np
import pandas as pd


def contain_nan(el):
    try:
        for e in el:
            if contain_nan(e):
                return True
    except TypeError:
        return np.isnan(el)
    return False


def bkm_quote_to_probas(bkm_quotes, outcomes_labels=None):
    if outcomes_labels is None: outcomes_labels = ['W', 'D', 'L']
    probabilities_with_fees = np.reciprocal(bkm_quotes[[lbl for lbl in outcomes_labels]].values)
    row_sums = probabilities_with_fees.sum(axis=1).reshape((-1, 1))
    probabilities = np.divide(probabilities_with_fees, row_sums)
    return probabilities


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
    pass
    # test_split()
    # test_get_last_matches()
    # test_create_full_fable()
