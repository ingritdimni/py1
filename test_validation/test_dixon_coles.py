import numpy as np
import pandas as pd
from data_preparation import first_data_preparation, create_dict_involved_teams
from dixoncoles_model import DixonColes

from my_utils import match_outcomes_hot_vectors, analyze_predictions
from utils_generic import contain_nan
from invest_strategies import ConstantAmountInvestStrategy, KellyInvestStrategy, ConstantStdDevInvestStrategy, \
    ConstantPercentInvestStrategy
from datetime import datetime
from utils_generic import printv

VERBOSE = 0


def test_case_dixon_coles_one_country_predictions():

    player_data, player_stats_data, team_data, match_data = first_data_preparation()
    countries = ['France', ]
    # countries = ['England', ]
    min_date = datetime.strptime('2016-04-30', "%Y-%m-%d").date()

    mask_countries = match_data['league_country'].isin(countries)
    match_data = match_data[mask_countries]
    # input(match_data['league_country'].unique())

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
    printv(1, VERBOSE, len(team_universe), team_universe)
    printv(1, VERBOSE, 'nb matches', match_data.shape[0])

    # save full_history before selecting recent matches to predict
    full_history = match_data

    mask_date = match_data['date'] >= min_date
    match_data = match_data[mask_date]

    exp_weight_fct = lambda t1, t2: np.exp(- 0.3 * (t2-t1).days / 365.25)
    model = DixonColes(team_universe, weight_fct=exp_weight_fct)
    printv(1, VERBOSE, " ... fit dixon coles parameters and predict match outcomes ... ")
    predictions, param_histo = model.fit_and_predict(match_data, full_history, nb_obs_years=1, verbose=VERBOSE,
                                                     home_goals_key='home_team_goal', away_goals_key='away_team_goal')
    printv(1, VERBOSE, " ... match outcomes predicted ... ")

    match_outcomes = match_outcomes_hot_vectors(match_data)
    bkm_quotes = pd.DataFrame()
    bkm_quotes['W'], bkm_quotes['D'], bkm_quotes['L'] = match_data['B365H'], match_data['B365D'], match_data['B365A']

    analysis = analyze_predictions(match_outcomes, predictions, bkm_quotes, verbose=VERBOSE, nb_max_matchs_displayed=40,
                                   fully_labelled_matches=match_data)
    model_log_loss, model_rps, (log_loss_comparison_l, rps_comparison_l) = analysis

    remove_nan_mask = [not contain_nan(bkm_quotes.iloc[i]) for i in range(bkm_quotes.shape[0])]
    bkm_quotes_r = bkm_quotes.iloc[remove_nan_mask]
    match_outcomes_r = match_outcomes.iloc[remove_nan_mask]
    predictions_r = predictions[remove_nan_mask]

    constant_invest_stgy = ConstantAmountInvestStrategy(1.)  # invest 1 in each match (if expected return > 1% actually)
    constant_sigma_invest_stgy = ConstantStdDevInvestStrategy(0.01)  # stdDev of each bet is 1% of wealth
    kelly_invest_stgy = KellyInvestStrategy()  # Kelly's ratio investment to maximize's wealth long term return
    constant_percent_stgy = ConstantPercentInvestStrategy(0.01)  # invest 1% of money each time

    for invest_stgy in [constant_invest_stgy, constant_sigma_invest_stgy, kelly_invest_stgy, constant_percent_stgy]:
        printv(1, VERBOSE, "\n#### results for ", invest_stgy.__class__.__name__, "####")
        init_wealth = 100
        df_recap_stgy = invest_stgy.apply_invest_strategy(predictions_r, bkm_quotes_r, match_outcomes_r,
                                                          init_wealth=init_wealth)

        printv(1, VERBOSE, df_recap_stgy[['invested_amounts', 'exp_gain_amounts', 'gain_amounts']].sum())
        printv(1, VERBOSE, 'wealth: from', init_wealth, 'to', round(df_recap_stgy['wealth'].iloc[-1], 4))


def test_case_optimize_parameters():

    player_data, player_stats_data, team_data, match_data = first_data_preparation()
    countries = ['France', ]
    min_date = '2013-07-31'
    max_date = '2014-07-31'
    match_data = match_data.loc[match_data['league_country'].isin(countries)]
    match_data = match_data[match_data['date'] >= min_date]
    match_data = match_data[match_data['date'] < max_date]

    mask_home = team_data['team_api_id'].isin(match_data['home_team_api_id'])
    mask_away = team_data['team_api_id'].isin(match_data['away_team_api_id'])
    team_universe = list(team_data[mask_home | mask_away]['team_long_name'])
    # print(len(team_universe), team_universe)
    # print('nb matches', match_data.shape[0])

    # on the below: non effective way to use team names as id (easier for human-checking and debugging)
    team_id_to_name, team_name_to_id = create_dict_involved_teams(match_data, team_data)
    match_data['home_team_id'] = match_data.apply(lambda x: team_id_to_name[x['home_team_api_id']], axis=1)
    match_data['away_team_id'] = match_data.apply(lambda x: team_id_to_name[x['away_team_api_id']], axis=1)

    model = DixonColes(team_universe)
    # print("... fit dixon coles parameters ...")

    opti_params = model.optimize_parameters(match_data, max_date, verbose=VERBOSE, home_goals_key='home_team_goal',
                                            away_goals_key='away_team_goal')
    # model.print_params(opti_params)

if __name__ == '__main__':
    test_case_optimize_parameters()
    test_case_dixon_coles_one_country_predictions()
