import numpy as np
import pandas as pd


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
