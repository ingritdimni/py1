import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def contain_nan(el):
    try:
        for e in el:
            if contain_nan(e):
                return True
    except TypeError:
        return np.isnan(el)
    return False


def get_values(array):
    if isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
        return array.values
    if isinstance(array, np.ndarray):
        return array
    raise TypeError('array should be pd.DataFrame or pd.Series or np.ndarray')


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

    y_dummy = match_outcomes_hot_vectors(match_results)
    return x_dummy, y_dummy


def rank_prob_score(predictions, y_hot_vectors):
    """ personnal implementation of ranked probability score (see Constantinou et al.)"""
    pred = get_values(predictions)
    y = get_values(y_hot_vectors)
    n, n_cat = pred.shape[0], pred.shape[1]
    assert(n == y.shape[0] and n_cat == y.shape[1])
    score = 0
    for i in range(n):
        for j in range(n_cat):
            score += (np.sum(predictions[i, :j+1]) - np.sum(y[i, :j+1]))**2
    return score/(n_cat - 1)


def match_outcomes_hot_vectors(match_results, home_goals_key='home_team_goal', away_goal_key='away_team_goal'):
    y_data = match_results.apply(lambda x: get_match_label(x, home_goals_key, away_goal_key), axis=1)
    y_dummy = pd.get_dummies(y_data, prefix_sep='')
    y_dummy = y_dummy[['W', 'D', 'L']]  # change order to get win first
    return y_dummy


def match_outcomes_indices(match_results, home_goals_key='home_team_goal', away_goal_key='away_team_goal'):
    y_indices = match_results.apply(lambda x: get_index_match_label(x, home_goals_key, away_goal_key), axis=1)
    return y_indices


def display_shapes(X_train, X_val, Y_train, Y_val):
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("Y_train shape:", Y_train.shape)
    print("Y_val shape:", Y_val.shape)


def display_results_analysis(y, pred, bkm_quotes, nb_max_matchs_displayed=10, compare_to_dummy_pred=True,
                             fully_labelled_matches=None, rank_prob_score_comparison=True, home_team_key='home_team_id',
                             away_team_key='away_team_id', home_goals_key = 'home_team_goal',
                             away_goal_key='away_team_goal'):
    assert(y.shape[0] == pred.shape[0] == bkm_quotes.shape[0])
    if fully_labelled_matches is not None:
        assert(y.shape[0] == fully_labelled_matches.shape[0])

    bkm_probas = bkm_quote_to_probas(bkm_quotes)
    if nb_max_matchs_displayed: print("--- on the below, few prediction examples")
    for i in range(min(y.shape[0], nb_max_matchs_displayed)):
        print()
        if fully_labelled_matches is not None:
            print(fully_labelled_matches[home_team_key].iloc[i], '  ', fully_labelled_matches[home_goals_key].iloc[i],
                  fully_labelled_matches[away_goal_key].iloc[i], '  ', fully_labelled_matches[away_team_key].iloc[i])
        print('model predictions :', pred[i])
        print('bkm probas:', list(bkm_probas[i]))
        print('bkm quote:', list(bkm_quotes.iloc[i]))
    print()
    print("total model log loss score:", round(log_loss(y, pred), 4))
    remove_nan_mask = [not contain_nan(bkm_probas[i]) for i in range(bkm_probas.shape[0])]
    print("bkm log loss score                           :", round(log_loss(y.iloc[remove_nan_mask],
                                                                           bkm_probas[remove_nan_mask]), 4))
    print("model log loss score on matches with bkm data:", round(log_loss(y.iloc[remove_nan_mask],
                                                                           pred[remove_nan_mask]), 4))
    if rank_prob_score_comparison:
        bkm_rps = rank_prob_score(bkm_probas[remove_nan_mask], y.iloc[remove_nan_mask])
        model_rps = rank_prob_score(pred[remove_nan_mask], y.iloc[remove_nan_mask])
        print("bkm rps score (on matches with bkm data):", bkm_rps)
        print("model rps score on matches with bkm data:", model_rps)
    if compare_to_dummy_pred:
        print("score of equiprobability prediction :", round(log_loss(y, np.full(y.shape, 1./3)), 4))


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
