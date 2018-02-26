import numpy as np
import copy
import pandas as pd

DATA_PATH = "D:/Football_betting/artificial_data/"


def main():
    nb_teams = 18
    nb_seasons = 10
    past_results, actual_team_params = create_minimalist_match_results(nb_teams, nb_seasons, seed=0)
    print(past_results.tail(20))
    past_results.to_csv(DATA_PATH + "poisson_results.csv")


def create_minimalist_match_results(nb_teams, nb_seasons, seed=0):
    """ Creates nb_teams teams with different poisson params representing their ability to score.
    Then simulates nb_seasons seasons, knowing that in each season each team plays against each other twice"""
    np.random.seed(seed)

    teams_params = create_teams(nb_teams)

    seasons_calendars = dict()
    base_calendar = None  # trick to avoid useless long computations
    for s in range(nb_seasons):
        seasons_calendars[s], base_calendar = create_calendar(nb_teams, base_calendar=base_calendar)

    df = pd.DataFrame(columns=['season', 'stage', 'home_team_goal', 'away_team_goal', 'home_team_id', 'away_team_id'])
    for s in range(nb_seasons):
        for d in range(len(seasons_calendars[s])):
            matches = seasons_calendars[s][d]
            for home_i, away_i in matches:
                home_param = teams_params[home_i]
                away_param = teams_params[away_i]
                home_goals, away_goals = play_match(home_param, away_param)
                df = df.append({'season': s+1, 'stage': d+1, 'home_team_goal': home_goals, 'away_team_goal': away_goals,
                                'home_team_id': home_i, 'away_team_id': away_i}, ignore_index=True)

    return df, teams_params


def play_match(home_param, away_param, seed=None):
    if seed: np.random.seed(seed)
    home_goals = np.random.poisson(home_param)
    away_goals = np.random.poisson(away_param)
    return home_goals, away_goals


def create_teams(nb_teams, param_min=0.8, param_max=2.5):
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
    if seed:
        np.random.seed(seed)
    init_keys = list(base_calendar.keys())
    modified_keys = list(init_keys)
    np.random.shuffle(modified_keys)
    my_calendar = dict()
    for i in range(len(base_calendar)):
        my_calendar[init_keys[i]] = base_calendar[modified_keys[i]]

    # second part of season is copied from 1rst part
    for d in range(nb_teams-1):
        my_calendar[nb_teams+d-1] = list(reversed(my_calendar[nb_teams-(d+1)-1]))

    return my_calendar, base_calendar


if __name__ == "__main__":
    main()
