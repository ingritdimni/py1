import numpy as np
import pandas as pd
from football_betting.weights import exp_weights, linear_gated_weights, season_count_fraction


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
        #print(s, d, season_count_fraction(s, d, cur_season, cur_day, nb_match_per_season))
        assert(abs(season_count_fraction(s, d, cur_season, cur_day, nb_match_per_season) - expected_res < epsilon))
