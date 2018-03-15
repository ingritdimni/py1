import numpy as np
import pandas as pd
from data_preparation import first_data_preparation, create_dict_involved_teams
from dixoncoles_model import dixon_coles_predictions
from my_utils import match_outcomes_hot_vectors, analyze_predictions
from utils_generic import contain_nan
from invest_strategies import ConstantAmountInvestStrategy, KellyInvestStrategy, ConstantStdDevInvestStrategy, \
    ConstantPercentInvestStrategy
from time import time
from datetime import datetime
from utils_generic import printv


def test_dixon_coles_multi_countries_prediction():
    player_data, player_stats_data, team_data, match_data = first_data_preparation()
    countries = ['France', 'England', 'Germany', 'Spain', 'Italy']

    # filter by countries
    mask_countries = match_data['league_country'].isin(countries)
    match_data = match_data.loc[mask_countries]

    # convert date input string to actual python date
    match_data['date'] = match_data.apply(lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M:%S").date(), axis=1)

    # on the below: non effective way to use team names as id (easier for human-checking and debugging)
    team_id_to_name, team_name_to_id = create_dict_involved_teams(match_data, team_data)
    match_data['home_team_id'] = match_data.apply(lambda x: team_id_to_name[x['home_team_api_id']], axis=1)
    match_data['away_team_id'] = match_data.apply(lambda x: team_id_to_name[x['away_team_api_id']], axis=1)

    # save full match history
    full_history = match_data

    # filter on recent matchs only (to make predictions on them)
    min_date = datetime.strptime('2013-07-31', "%Y-%m-%d").date()
    mask_date = match_data['date'] >= min_date
    matches_to_predict = match_data[mask_date]

    print("nb matches full history", full_history.shape[0])
    print("nb matches_to_predict", matches_to_predict.shape[0])

    # define weight fct (default is one)
    dixon_coles_params = {'weight_fct': lambda t1, t2: np.exp(- 0.3 * (t2 - t1).days / 365.25)}
    predictions = dixon_coles_predictions(matches_to_predict, full_history, dixon_coles_params=dixon_coles_params,
                                          verbose=1)

#
# def dixon_coles_predictions(matches_to_predict, full_match_history, dixon_coles_params=None, verbose=1,
#                             intermediary_analysis=True, home_team_key='home_team_id',
#                             away_team_key='away_team_id', home_goals_key='home_team_goal',
#                             away_goals_key='away_team_goal', time_key='date', season_key='season', stage_key='stage'):
#
#     # default model params
#     if dixon_coles_params is None:
#         dixon_coles_params = {'weight_fct': lambda t1, t2: np.exp(- 0.3 * (t2 - t1).days / 365.25)}
#     # if dixon_coles_params is None:
#     #     dixon_coles_params = dict()
#
#     # create an index to be able to return predictions in the order of the input (not the order it s been computed)
#     matches_to_predict['tmp_index'] = range(len(matches_to_predict))
#     countries = list(matches_to_predict['league_country'].unique())
#     all_predictions = None
#     for country in countries:
#         printv(1, verbose, "\n ####  WORKING WITH DATA FROM", country, " #### ")
#         match_data = matches_to_predict[matches_to_predict['league_country'].isin([country, ])]
#         match_history = full_match_history[full_match_history['league_country'].isin([country, ])]
#
#         # on the below: define our team universe (teams we calibrate parameters on)
#         team_universe = set(match_history[home_team_key].unique()) | set(match_history[away_team_key].unique())
#         printv(1, verbose, ' ...', len(team_universe), ' teams involved:', *team_universe, '...')
#         printv(1, verbose, ' ...', match_data.shape[0], 'matches to predict ...')
#
#         model = DixonColes(team_universe, **dixon_coles_params)
#         printv(1, verbose, " ... fit dixon coles parameters and predict match outcomes ... ")
#         predictions, param_histo = model.fit_and_predict(match_data, match_history, nb_obs_years=3,
#                                                          verbose=verbose, home_team_key=home_team_key,
#                                                          away_team_key=away_team_key, home_goals_key=home_goals_key,
#                                                          away_goals_key=away_goals_key, time_key=time_key,
#                                                          season_key=season_key, stage_key=stage_key)
#         printv(1, verbose, " ... match outcomes predicted ... ")
#
#         if len(countries) > 1 and intermediary_analysis:  # either we display intermediary predictions quality or not
#             match_outcomes = match_outcomes_hot_vectors(match_data)
#             bkm_quotes = pd.DataFrame()
#             bkm_quotes['W'], bkm_quotes['D'], bkm_quotes['L'] = match_data['B365H'], match_data['B365D'], match_data[
#                 'B365A']
#             analysis = analyze_predictions(match_outcomes, predictions, bkm_quotes, nb_max_matchs_displayed=40,
#                                            fully_labelled_matches=match_data, verbose=verbose,
#                                            home_team_key=home_team_key, away_team_key=away_team_key,
#                                            home_goals_key=home_goals_key, away_goals_key=away_goals_key)
#
#             model_log_loss, model_rps, (log_loss_comparison_l, rps_comparison_l) = analysis
#
#         # add predictions to those already made
#         predictions_with_index = np.append(match_data['tmp_index'].values.reshape((-1, 1)), predictions, axis=1)
#         if all_predictions is not None:
#             all_predictions = np.append(all_predictions, predictions_with_index, axis=0)
#         else:
#             all_predictions = predictions_with_index
#
#     # exctract all predictions, resort them by their index, and remove the index
#     all_predictions = all_predictions[all_predictions[:, 0].argsort()][:, 1:]
#
#     # perform a global analysis
#     all_match_outcomes = match_outcomes_hot_vectors(matches_to_predict)
#     all_bkm_quotes = pd.DataFrame()
#     all_bkm_quotes['W'] = matches_to_predict['B365H']
#     all_bkm_quotes['D'] = matches_to_predict['B365D']
#     all_bkm_quotes['L'] = matches_to_predict['B365A']
#     analysis = analyze_predictions(all_match_outcomes, all_predictions, all_bkm_quotes, nb_max_matchs_displayed=40,
#                                    fully_labelled_matches=matches_to_predict, verbose=verbose,
#                                    home_team_key=home_team_key, away_team_key=away_team_key,
#                                    home_goals_key=home_goals_key, away_goals_key=away_goals_key)
#     print("final_pred shape", all_predictions.shape)
#     return all_predictions

if __name__ == '__main__':
    test_dixon_coles_multi_countries_prediction()
