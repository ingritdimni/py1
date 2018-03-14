import numpy as np
import pandas as pd
from data_preparation import first_data_preparation, create_dict_involved_teams
from dixoncoles_model import DixonColes
from my_utils import contain_nan, match_outcomes_hot_vectors, display_results_analysis
from invest_strategies import ConstantAmountInvestStrategy, KellyInvestStrategy, ConstantStdDevInvestStrategy, \
    ConstantPercentInvestStrategy
from time import time
from datetime import datetime


def dixon_coles_full_predictions():

    start_time = time()

    player_data, player_stats_data, team_data, match_data = first_data_preparation()
    # countries = ['France', ]
    countries = ['England', ]
    min_date = datetime.strptime('2013-07-31', "%Y-%m-%d").date()

    mask_countries = match_data['league_country'].isin(countries)
    match_data = match_data.loc[mask_countries]

    # convert date input string to actual python date
    match_data['date'] = match_data.apply(lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M:%S").date(), axis=1)

    # on the below: non effective way to use team names as id (easier for human-checking and debugging)
    team_id_to_name, team_name_to_id = create_dict_involved_teams(match_data, team_data)
    match_data['home_team_id'] = match_data.apply(lambda x: team_id_to_name[x['home_team_api_id']], axis=1)
    match_data['away_team_id'] = match_data.apply(lambda x: team_id_to_name[x['away_team_api_id']], axis=1)

    # on the below: we define our team universe (teams we calibrate parameters on)
    mask_home = team_data['team_api_id'].isin(match_data['home_team_api_id'])
    mask_away = team_data['team_api_id'].isin(match_data['away_team_api_id'])
    team_universe = list(team_data[mask_home | mask_away]['team_long_name'])
    print(len(team_universe), team_universe)
    print('nb matches', match_data.shape[0])

    # save full_history before selecting recent matches to predict
    full_history = match_data

    mask_date = match_data['date'] >= min_date
    match_data = match_data[mask_date]

    exp_weight_fct = lambda t1, t2: np.exp(- 0.3 * (t2-t1).days / 365.25)
    model = DixonColes(team_universe, weight_fct=exp_weight_fct)
    print(" ... fit dixon coles parameters and predict match outcomes ... ")
    predictions, param_histo = model.fit_and_predict(match_data, full_history, nb_obs_years=3, verbose=2,
                                                     home_goals_key='home_team_goal', away_goals_key='away_team_goal')
    print(" ... match outcomes predicted ... ")
    print(" ... so far, computations made in", time() - start_time, "seconds ...")

    match_outcomes = match_outcomes_hot_vectors(match_data)
    bkm_quotes = pd.DataFrame()
    bkm_quotes['W'], bkm_quotes['D'], bkm_quotes['L'] = match_data['B365H'], match_data['B365D'], match_data['B365A']

    display_results_analysis(match_outcomes, predictions, bkm_quotes, nb_max_matchs_displayed=40,
                             fully_labelled_matches=match_data)

    remove_nan_mask = [not contain_nan(bkm_quotes.iloc[i]) for i in range(bkm_quotes.shape[0])]
    bkm_quotes_r = bkm_quotes.iloc[remove_nan_mask]
    match_outcomes_r = match_outcomes.iloc[remove_nan_mask]
    predictions_r = predictions[remove_nan_mask]

    constant_invest_stgy = ConstantAmountInvestStrategy(1.)  # invest 1 in each match (if expected return > 1% actually)
    constant_sigma_invest_stgy = ConstantStdDevInvestStrategy(0.01)  # stdDev of each bet is 1% of wealth
    kelly_invest_stgy = KellyInvestStrategy()  # Kelly's ratio investment to maximize's wealth long term return
    constant_percent_stgy = ConstantPercentInvestStrategy(0.01)  # invest 1% of money each time

    for invest_stgy in [constant_invest_stgy, constant_sigma_invest_stgy, kelly_invest_stgy, constant_percent_stgy]:
        print("\n#### results for ", invest_stgy.__class__.__name__, "####")
        init_wealth = 100
        df_recap_stgy = invest_stgy.apply_invest_strategy(predictions_r, bkm_quotes_r, match_outcomes_r,
                                                          init_wealth=init_wealth)

        print(df_recap_stgy[['invested_amounts', 'exp_gain_amounts', 'gain_amounts']].sum())
        print('wealth: from', init_wealth, 'to', round(df_recap_stgy['wealth'].iloc[-1], 4))


def dixon_coles_opti_param():
    # min_date = '2013-06-31'
    # ma_date = '2013-07-14'
    # input(ma_date-min_date)
    # print(match_data['date'].iloc[0], match_data['date'].iloc[0].__class__)
    # d1 = datetime.datetime.strptime(match_data['date'].iloc[0], "%Y-%m-%d %H:%M:%S").date()
    # d2 = datetime.datetime.strptime(match_data['date'].iloc[100], "%Y-%m-%d %H:%M:%S").date()
    # d3 = datetime.datetime.strptime(match_data['date'].iloc[1000], "%Y-%m-%d %H:%M:%S").date()
    # print(d1, d2, d1.__class__, (d1-d2).days, int((d3-d1).days)/365.25)
    # input()

    player_data, player_stats_data, team_data, match_data = first_data_preparation()
    countries = ['France', ]
    min_date = '2013-06-31'
    match_data = match_data.loc[match_data['league_country'].isin(countries)]
    match_data = match_data[match_data['date'] >= min_date]

    mask_home = team_data['team_api_id'].isin(match_data['home_team_api_id'])
    mask_away = team_data['team_api_id'].isin(match_data['away_team_api_id'])
    team_universe = list(team_data[mask_home | mask_away]['team_long_name'])
    print(len(team_universe), team_universe)
    print('nb matches', match_data.shape[0])

    # on the below: non effective way to use team names as id (easier for human-checking and debugging)
    team_id_to_name, team_name_to_id = create_dict_involved_teams(match_data, team_data)
    match_data['home_team_id'] = match_data.apply(lambda x: team_id_to_name[x['home_team_api_id']], axis=1)
    match_data['away_team_id'] = match_data.apply(lambda x: team_id_to_name[x['away_team_api_id']], axis=1)

    model = DixonColes(team_universe)
    print("... fit dixon coles parameters ...")

    opti_params = model.optimize_parameters(match_data, home_goals_key='home_team_goal',
                                            away_goals_key='away_team_goal')
    model.display_params(opti_params)

if __name__ == '__main__':
    dixon_coles_full_predictions()
