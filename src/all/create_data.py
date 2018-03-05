import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from poisson import PoissonHelper


DATA_PATH = "D:/Football_betting/artificial_data/"  # default export path, might be overwritten
DISPLAY_GRAPH = True


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

    display_param_history(params_history, nb_diffused_seasons)


def test_export_stationary_poisson_match_results(path=DATA_PATH):
    """" exports stationary poisson match results and probabilities """
    nb_teams = 18
    nb_seasons = 10
    df_results, df_probas, actual_team_params = create_stationary_poisson_match_results(nb_teams, nb_seasons, seed=0)
    print(df_results.tail(20))
    print(df_probas.tail(20))
    df_results.to_csv(path + "stationary_poisson_results.csv", index=False)
    df_probas.to_csv(path + "stationary_poisson_results_probabilities.csv", index=False)


def test_export_dynamic_poisson_match_results(path=DATA_PATH):
    """" exports dynamic poisson match results and probabilities """
    nb_teams = 18
    nb_seasons = 10
    df_results, df_probas, params_history = create_dynamic_poisson_match_results(nb_teams, nb_seasons,
                                                                                 nb_fixed_seasons=1, seed=0)
    print(df_results.tail(20))
    print(df_probas.tail(20))
    df_results.to_csv(path + "dynamic_poisson_results.csv", index=False)
    df_probas.to_csv(path + "dynamic_poisson_results_probabilities.csv", index=False)

    display_param_history(params_history, nb_seasons)


def test_create_bookmaker_quotes():
    match_probas = pd.DataFrame({'W': [0.4, 0.6, 0.1, 1./3], 'D': [0.2, 0.2, 0.5, 1./3], 'L': [0.4, 0.2, 0.4, 1./3]})
    quotes = create_noisy_bookmaker_quotes(match_probas, seed=0)
    print(quotes)
    print(quotes.apply(lambda x: np.reciprocal(x)).sum(axis=1))


def create_stationary_poisson_match_results(nb_teams, nb_seasons, param_min=0.7, param_max=2.75, seed=None,
                                            export=False, export_path=DATA_PATH):
    """ Creates nb_teams teams with different poisson params representing their ability to score.
    Then simulates nb_seasons seasons, knowing that in each season each team plays against each other twice"""
    if seed is not None: np.random.seed(seed)
    teams_params = create_teams(nb_teams, param_min, param_max)

    seasons_calendars = dict()
    base_calendar = None  # trick to avoid useless long computations
    for s in range(nb_seasons):
        seasons_calendars[s], base_calendar = create_calendar(nb_teams, base_calendar=base_calendar)

    df_results = pd.DataFrame(columns=['season', 'stage', 'home_team_goal', 'away_team_goal', 'home_team_id',
                                       'away_team_id'])
    df_probas = pd.DataFrame(columns=['W', 'D', 'L'])
    for s in range(nb_seasons):
        for d in range(len(seasons_calendars[s])):
            matches = seasons_calendars[s][d]
            for home_i, away_i in matches:
                home_param = teams_params[home_i]
                away_param = teams_params[away_i]
                home_goals, away_goals = PoissonHelper.play_match(home_param, away_param)
                df_results = df_results.append({'season': s+1, 'stage': d+1, 'home_team_goal': home_goals,
                                                'away_team_goal': away_goals, 'home_team_id': home_i,
                                                'away_team_id': away_i}, ignore_index=True)
                win, draw, loss = perfect_prediction(home_i, away_i, teams_params)
                df_probas = df_probas.append({'W': win, 'D': draw, 'L': loss}, ignore_index=True)
    assert(df_probas.shape[0] == df_results.shape[0])
    if export:
        param_str = 't' + str(nb_teams) + '_s' + str(nb_seasons) + '_'
        df_results.to_csv(export_path + param_str + "stationary_poisson_results.csv", index=False)
        df_probas.to_csv(export_path + param_str + "stationary_poisson_results_probabilities.csv", index=False)
    return df_results, df_probas, teams_params


def create_dynamic_poisson_match_results(nb_teams, nb_seasons, param_min=0.6, param_max=3., nb_jumps_start_season=10,
                                         nb_fixed_seasons=0, update_param_a=0.02, update_param_b=0.01,
                                         update_avg_param=1.7, seed=None, export=False, export_path=DATA_PATH):
    """ Creates nb_teams teams with different poisson params representing their ability to score.
    Then simulates nb_seasons seasons, knowing that in each season each team plays against each other twice
    uses poisson param dynamic as described in update_teams_params """
    assert(nb_fixed_seasons <= nb_seasons)  # params do not move during nb_fixed seasons starting from the end
    if seed is not None: np.random.seed(seed)
    teams_params = create_teams(nb_teams, param_min, param_max)

    seasons_calendars = dict()
    base_calendar = None  # trick to avoid useless long computations
    for s in range(nb_seasons):
        seasons_calendars[s], base_calendar = create_calendar(nb_teams, base_calendar=base_calendar)

    df_results = pd.DataFrame(columns=['season', 'stage', 'home_team_goal', 'away_team_goal', 'home_team_id',
                                       'away_team_id'])
    df_probas = pd.DataFrame(columns=['W', 'D', 'L'])
    params_history = {t: list() for t in range(nb_teams + 1)[1:]}  # excludes first params
    for s in range(nb_seasons):
        for step in range(nb_jumps_start_season):  # teams change at interseason --> more volatility ?!
            teams_params = update_teams_params(teams_params, update_param_a, update_param_b, param_min, param_max,
                                               update_avg_param)
        for d in range(len(seasons_calendars[s])):

            # save current params
            for t in range(nb_teams+1)[1:]:
                params_history[t].append(teams_params[t])

            matches = seasons_calendars[s][d]
            for home_i, away_i in matches:

                home_param = teams_params[home_i]
                away_param = teams_params[away_i]
                home_goals, away_goals = PoissonHelper.play_match(home_param, away_param)
                df_results = df_results.append({'season': s+1, 'stage': d+1, 'home_team_goal': home_goals,
                                                'away_team_goal': away_goals, 'home_team_id': home_i,
                                                'away_team_id': away_i}, ignore_index=True)
                win, draw, loss = perfect_prediction(home_i, away_i, teams_params)
                df_probas = df_probas.append({'W': win, 'D': draw, 'L': loss}, ignore_index=True)
                if s < (nb_seasons - nb_fixed_seasons):  # if param must be updated
                    teams_params = update_teams_params(teams_params, update_param_a, update_param_b, param_min,
                                                       param_max, update_avg_param)  # update params
    assert(df_probas.shape[0] == df_results.shape[0])
    if export:
        param_str = 't' + str(nb_teams) + '_s' + str(nb_seasons) + '_'
        df_results.to_csv(export_path + param_str + "dynamic_poisson_results.csv", index=False)
        df_probas.to_csv(export_path + param_str + "dynamic_poisson_results_probabilities.csv", index=False)
    return df_results, df_probas, params_history


def display_param_history(params_history, nb_diffused_seasons=None):
    nb_teams = len(params_history)
    fig, ax = plt.subplots(1, 1)
    for t in range(nb_teams+1)[1:]:
        ax.plot(params_history[t], label="param_team_" + str(t))
    if nb_diffused_seasons:  # to add vertical line between seasons (optional)
        for s in range(nb_diffused_seasons)[1:]:
            plt.axvline(x=s * (nb_teams-1) * 2)
    #legend = ax.legend(loc='upper left', shadow=True)

    if DISPLAY_GRAPH: plt.show()


def update_teams_params(teams_params, a=0.02, b=0.01, min_param=0.6, max_param=3., target_sum=1.7, seed=None):
    """ creates a new set of params according to formula: new_param := noise * (a * old_param + b).
    teams_params input is a dictionary of params. returns a dictionary of updated params"""
    if seed is not None: np.random.seed(seed)
    noise = np.random.randn(len(teams_params))
    all_teams = list(teams_params.keys())
    nb_teams = len(all_teams)
    avg, new_params = 0, dict()

    # random step in param
    for i in range(nb_teams):
        new_params[all_teams[i]] = teams_params[all_teams[i]] + noise[i] * (a * teams_params[all_teams[i]] + b)
        avg += new_params[all_teams[i]] / nb_teams

    # normalization ratio of teams params (to avoid all very strong or very weak)
    ratio = target_sum / avg if target_sum else 1.

    # normalization attempt + respect of bounds anyway
    for i in range(nb_teams):
        new_params[all_teams[i]] = max(min(new_params[all_teams[i]] * ratio, max_param), min_param)

    return new_params


def create_noisy_bookmaker_quotes(match_probas, std_dev=0.05, fees=0.05, seed=None):
    """ create bookmaker quotes from actual probas.
    add gaussian noise around actual proba, then convert it to bookmaker quotes adding some fees
    TODO: improve, as fees might be different than expected, if p = init_p + std_dev * noise is not between [0, 1]"""
    if seed is not None: np.random.seed(seed)
    noise = np.random.randn(*match_probas.shape)
    noisy_probas = np.clip(match_probas + std_dev * noise, 10e-7, 1.)
    row_sums = noisy_probas.sum(axis=1).to_frame()  # output is a Series, need to cast it to broadcast in next line
    noisy_probas_with_fees = np.clip(np.divide(np.multiply(noisy_probas, 1. + fees), row_sums), 10e-7, 1.)
    noisy_probas_with_fees = noisy_probas_with_fees[['W', 'D', 'L']]  # to get columns in order
    return np.reciprocal(noisy_probas_with_fees)


def perfect_prediction(team_home, team_away, teams_param):
    """ returns perfect prediction if match has been played considering poisson distrib for scoring (see play match)"""
    return PoissonHelper.match_outcomes_probabilities(teams_param[team_home], teams_param[team_away])


def create_teams(nb_teams, param_min=0.7, param_max=2.75):
    """ create a dictionary of nb_teams teams, where key is int from 1 to nb_team, and value is associated param.
    Params are chosen regularly uniformly within given range"""
    # create team names
    teams = [i+1 for i in range(nb_teams)]

    # create team parameters
    param_step = (param_max - param_min) / (nb_teams - 1.)
    team_params = [param_max - i * param_step for i in range(nb_teams)]

    return dict(zip(teams, team_params))


def create_base_calendar(nb_teams):
    """ this function creates a season calendar.
    For now, it does not take into account home or away matches, but i might be improved easily"""
    assert nb_teams % 2 == 0

    all_calendars = list()

    def played_together(calendar, t1, t2):
        for j in calendar.keys():
            for m in calendar[j]:
                if m[0] == t1 and m[1] == t2:
                    return True
        return False

    def rec_fct(j, calendar, all_good_calendars):
        if j == nb_teams - 1:  # good calendar has been found
            all_good_calendars.append(calendar)
            raise LookupError

        # find already planned match for on going day
        l, max_t = list(),  -1
        if j in calendar.keys():
            for m in calendar[j]:
                l.append(m[0])
                l.append(m[1])
                max_t = max(max_t, m[0])
        else:  # initialization of new day
            calendar[j] = list()
        remaining_team_indices = set(range(nb_teams+1)[1:]) - set(l)

        for t1 in remaining_team_indices:
            for t2 in remaining_team_indices:
                min_t1t2, max_t1t2 = min(t1, t2), max(t1, t2)
                if max_t < t1 < t2 and not played_together(calendar, min_t1t2, max_t1t2):
                    new_calendar = copy.deepcopy(calendar)
                    new_calendar[j].append([min_t1t2, max_t1t2])
                    if len(new_calendar[j]) * 2. == nb_teams:
                        rec_fct(j+1, new_calendar, all_good_calendars)
                    else:
                        rec_fct(j, new_calendar, all_good_calendars)

    # this algo is actually way too heavy, so we choose to stop it on first solution found
    try:
        rec_fct(0, dict(), all_calendars)
    except LookupError:
        obtained_calendar = all_calendars[0]

    return obtained_calendar


def create_calendar(nb_teams, base_calendar=None, seed=None):

    if not base_calendar:
        base_calendar = create_base_calendar(nb_teams)

    # on teh below, we shuffle results
    if seed is not None: np.random.seed(seed)
    init_keys = list(base_calendar.keys())
    modified_keys = list(init_keys)
    np.random.shuffle(modified_keys)
    my_calendar = dict()
    for i in range(len(base_calendar)):
        my_calendar[init_keys[i]] = base_calendar[modified_keys[i]]

    # second part of season is copied from 1rst part
    for d in range(nb_teams-1):
        symetric_matches = my_calendar[nb_teams - (d + 1) - 1]
        my_calendar[nb_teams+d-1] = [[m[1], m[0]] for m in symetric_matches]

    return my_calendar, base_calendar


if __name__ == "__main__":
    test_create_bookmaker_quotes()
    #test_param_dynamic()
    #test_export_dynamic_poisson_match_results()
    #export_stationary_poisson_match_results()
