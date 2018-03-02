import numpy as np
import pandas as pd


def test_weights():
    nb_teams = 20
    nb_match_per_season = (nb_teams - 1) * 2
    cur_season = 10
    cur_day = 17

    tested_inputs = [[4, 1], [4, 27], [6, 15], [7, 31], [9, 13], [9, 17], [9, 18], [10, 2], [10, 17]]
    df_inputs = pd.DataFrame(columns=['season', 'stage'])
    for s, d in tested_inputs:
        df_inputs = df_inputs.append({'season': s, 'stage': d}, ignore_index=True)

    df_exp_weights = exp_weights(df_inputs, cur_season, cur_day, nb_match_per_season, season_rate=0.10,
                                 season_label='season', day_label='stage')
    df_linear_weights = linear_gated_weights(df_inputs, cur_season, cur_day, nb_match_per_season, nb_seasons_to_keep=5,
                                             season_label='season', day_label='stage')

    # print(df_exp_weights)
    # print(df_linear_weights)
    for df_weights in [df_exp_weights, df_linear_weights]:
        l = list(df_weights)
        assert(all(l[i] <= l[i + 1] for i in range(len(l) - 1)))


def test_season_count_fraction():
    nb_teams = 20
    nb_match_per_season = (nb_teams - 1) * 2
    cur_season = 10
    cur_day = 17
    epsilon = 10e-7

    tested_inputs = [[9, 17, 1.0], [9, 13, 1.105263157894737], [9, 18, 0.9736842105263158],
                     [10, 2, 0.39473684210526316], [10, 17, 0.]]
    for s, d, expected_res in tested_inputs:
        print(s, d, season_count_fraction(s, d, cur_season, cur_day, nb_match_per_season))
        assert(abs(season_count_fraction(s, d, cur_season, cur_day, nb_match_per_season) - expected_res < epsilon))


def exp_weights(inputs, current_season, current_day, nb_days_per_season, season_rate=0.10, normalize=False,
                season_label='season', day_label='stage'):

    def local_exp_weight(s, d):  # shorter version of exp weights using context variables
        return exp_weight(season_rate, season_count_fraction(s, d, current_season, current_day, nb_days_per_season))

    weights = inputs.apply(lambda x: local_exp_weight(x[season_label], x[day_label]), axis=1).to_frame()
    if normalize: weights = weights.divide(weights.mean())
    return weights


def linear_gated_weights(inputs, current_season, current_day, nb_days_per_season, nb_seasons_to_keep, normalize=False,
                         season_label='season', day_label='stage'):

    def linear_gated_weight(s, d):
        dt = season_count_fraction(s, d, current_season, current_day, nb_days_per_season)
        return max(min(1. - 1/nb_seasons_to_keep * dt, 1.), 0.)

    weights = inputs.apply(lambda x: linear_gated_weight(x[season_label], x[day_label]), axis=1).to_frame()
    if normalize: weights = weights.divide(weights.mean())
    return weights


def one_weights(n):
    return pd.DataFrame(1., index=np.arange(n), columns=['weight'])


def exp_weight(season_rate, season_count_fraction):
    """ compute weight seen as an exponential discount factor, i.e. exp(-r * t), with r and t inputs"""
    return np.exp(- season_rate * season_count_fraction)


def season_count_fraction(past_season, past_day, current_season, current_day, nb_days_per_season):
    """ computes season count fraction between past season and days to current season and day """
    total_seasons = current_season - past_season
    total_days = current_day - past_day
    if total_days < 0:
        total_days += nb_days_per_season
        total_seasons -= 1
    assert(total_seasons >= 0)
    assert(total_days >= 0)
    return (total_seasons * nb_days_per_season + total_days) / nb_days_per_season


if __name__ == "__main__":
    test_season_count_fraction()
    test_weights()
