import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data_preparation import first_data_preparation, create_dict_involved_teams
from poisson import PoissonHelper
from dateutil.relativedelta import relativedelta
from time import time
import datetime


def test_dixon_coles():
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


class DixonColes(object):

    def __init__(self, team_universe, weight_fct=None, padding_scored=0.8, padding_conceded=1.5,
                 padding_intervention_ratio=0.33):
        """ team universe is an iterable of all involved teams of a given studied universe. Might be IDs or names"""
        self.nb_teams = len(team_universe)
        self.nb_params = self.nb_teams * 2 + 1
        self.team_index_to_id = dict(zip(range(self.nb_teams), team_universe))
        self.team_id_to_index = dict(zip(team_universe, range(self.nb_teams)))
        assert(len(self.team_index_to_id) == len(self.team_id_to_index))
        if weight_fct is None:
            self.weight_fct = lambda t1, t2: 1.  # no weights
        else:
            self.weight_fct = weight_fct

        # if not enough data, we 'create' fake data with chosen properties to have more consistent results
        self.padding_scored = padding_scored
        self.padding_conceded = padding_conceded
        self.padding_intervention_ratio = padding_intervention_ratio

    def fit_and_predict(self, matches_to_predict, full_matches_history, nb_obs_years=3, padding=True, verbose=2,
                        home_team_key='home_team_id', away_team_key='away_team_id', home_goals_key='home_goals',
                        away_goals_key='away_goals', time_key='date', season_key='season', stage_key='stage'):

        sorted_matches = matches_to_predict.sort_values(by=[season_key, stage_key, time_key])

        # first parameters calibration
        pred_season = sorted_matches[season_key].iloc[0]
        pred_stage = sorted_matches[stage_key].iloc[0]
        pred_time = sorted_matches[time_key].iloc[0]
        if verbose >= 2:
            print("current calibration;   season", pred_season, "  day", pred_stage, "  time", pred_time)
        min_hist_time = pred_time - relativedelta(years=nb_obs_years)
        relevant_match_history = full_matches_history[min_hist_time <= full_matches_history[time_key]]
        relevant_match_history = relevant_match_history[relevant_match_history[time_key] < pred_time]
        opti_params = self.optimize_parameters(relevant_match_history, pred_time, padding=padding, verbose=verbose,
                                               home_team_key=home_team_key, away_team_key=away_team_key,
                                               home_goals_key=home_goals_key, away_goals_key=away_goals_key,
                                               time_key=time_key)
        if verbose >= 3:
            self.display_params(opti_params)
        params_history = [[pred_time, opti_params], ]

        # storage of outcomes probabilities
        df_predictions = pd.DataFrame(columns=['W', 'D', 'L'])

        for i, match in sorted_matches.iterrows():
            # parameters calibration
            cur_season = match[season_key]
            cur_stage = match[stage_key]
            cur_time = match[time_key]
            if cur_season != pred_season or cur_stage != pred_stage:
                if verbose >= 2:
                    print("current calibration;   season", cur_season, "  day", cur_stage, "  time", cur_time)
                pred_season, pred_stage, pred_time = cur_season, cur_stage, cur_time
                # below line assumes time column is a string, ugly !
                min_hist_time = pred_time - relativedelta(years=nb_obs_years)
                relevant_match_history = full_matches_history[min_hist_time <= full_matches_history[time_key]]
                relevant_match_history = relevant_match_history[relevant_match_history[time_key] < pred_time]
                # we start optimization by last most relevant params
                opti_params = self.optimize_parameters(relevant_match_history, pred_time, init_params=opti_params,
                                                       padding=padding, verbose=verbose, home_team_key=home_team_key,
                                                       away_team_key=away_team_key, home_goals_key=home_goals_key,
                                                       away_goals_key=away_goals_key, time_key=time_key)
                if opti_params is not None:
                    params_history.append([pred_time, opti_params])
                if verbose >= 3:
                    self.display_params(opti_params)

        # make prediction by finding adapted param and use it to predict outcome
        sorted_params_history = sorted(params_history, key=lambda x: x[0])
        for i, match in matches_to_predict.iterrows():

            # find adapted params for given match
            match_t = match[time_key]
            t, params_t = sorted_params_history[0]
            for next_t, params_next_t in sorted_params_history:
                if next_t > match_t:
                    break  # params to use have been found ! --> t, params_t
                params_t = params_next_t

            # predictions using params
            p_w, p_d, p_l = self.predict_match_outcome(match[home_team_key], match[away_team_key], params_t)
            df_predictions = df_predictions.append({'W': p_w, 'D': p_d, 'L': p_l}, ignore_index=True)
            if verbose >= 3:
                print("prediction;", match[home_team_key], match[away_team_key], " --> ", round(p_w, 4), round(p_d, 4),
                      round(p_l, 4))

        return df_predictions.values, sorted_params_history

    def predict_match_outcome(self, home_team_id, away_team_id, params):
        alpha_home = params[self.team_id_to_index[home_team_id]]
        beta_home = params[self.team_id_to_index[home_team_id] + self.nb_teams]
        alpha_away = params[self.team_id_to_index[away_team_id]]
        beta_away = params[self.team_id_to_index[away_team_id] + self.nb_teams]
        gamma = params[-1]

        lambda_ = alpha_home * beta_away * gamma
        mu_ = alpha_away * beta_home

        p_w, p_d, p_l = PoissonHelper.match_outcomes_probabilities(lambda_, mu_)
        return p_w, p_d, p_l

    def optimize_parameters(self, matches, current_time, init_params=None, padding=True, verbose=1,
                            home_team_key='home_team_id', away_team_key='away_team_id',
                            home_goals_key='home_goals', away_goals_key='away_goals',
                            time_key='date', control_dates=True):

        if control_dates: # control we calibrate params on past data
            assert(matches[time_key].max() < current_time)

        # init params
        if init_params is None:
            init_params = np.ones((self.nb_params, 1))

        bounds = ((0.2, 5),) * self.nb_params

        # define local functions involved in optimization
        def constraint_fct(params):  # avg of alphas and betas must be one
            return (np.sum(params[0:self.nb_teams]) - self.nb_teams) ** 2 + \
                   (np.sum(params[self.nb_teams:2*self.nb_teams]) - self.nb_teams)**2

        def constraint_fct_der(params):
            jac = np.zeros_like(params)
            alpha_cur = np.sum(params[0:self.nb_teams]) - self.nb_teams
            beta_cur = np.sum(params[self.nb_teams:2*self.nb_teams]) - self.nb_teams
            for i in range(self.nb_teams):
                jac[i] = 2. * params[i] * alpha_cur
            for i in range(self.nb_teams,  2*self.nb_teams):
                jac[i] = 2. * params[i] * beta_cur
            return jac

        def likelihood_m(params):
            return - self._likelihood(matches, params, current_time, padding=padding, home_team_key=home_team_key,
                                      away_team_key=away_team_key, home_goals_key=home_goals_key,
                                      away_goals_key=away_goals_key, time_key=time_key)

        def likelihood_jac_m(params):
            return - self._likelihood_jac(matches, params, current_time, padding=padding, home_team_key=home_team_key,
                                          away_team_key=away_team_key, home_goals_key=home_goals_key,
                                          away_goals_key=away_goals_key, time_key=time_key)

        # other ok methods; TNC or L-BFGS-B
        res = minimize(likelihood_m, init_params, jac=likelihood_jac_m, method='Newton-CG', bounds=bounds,
                       options={'xtol': 10e-3, 'disp': False},
                       constraints=({'type': 'eq', 'fun': constraint_fct, 'jac': constraint_fct_der},))
        if not res.success:
            if verbose:
                print(" fail to calibrate parameters with method Newton-CG. trying another method (TNC)")
            res = minimize(likelihood_m, init_params, jac=likelihood_jac_m, method='TNC', bounds=bounds,
                           options={'xtol': 10e-3, 'disp': False},
                           constraints=({'type': 'eq', 'fun': constraint_fct, 'jac': constraint_fct_der},))
            if not res.success:
                print('\033[91m' + "fail to calibrate parameters on date " + str(current_time) + '\033[0m')
                return None
        return res.x

    def display_params(self, params):
        print(" ----  DIXON COLES PARAMETERS  ---- ")
        for i in range(self.nb_teams):
            print(self.team_index_to_id[i], round(params[i], 4), round(params[i + self.nb_teams], 4))
        print('gamma:', round(params[-1], 4))
        print()

    def _likelihood(self, matches, params, current_time, padding=False, home_team_key='home_team_id',
                    away_team_key='away_team_id', home_goals_key='home_goals', away_goals_key='away_goals',
                    time_key='date'):
        # TODO : check matches are all in the past !
        nb_matches = matches.shape[0]
        result, total_weights = 0., 0.
        gamma = params[self.nb_params-1]

        total_weight_per_team = np.zeros(self.nb_teams)

        for i in range(nb_matches):
            home_team_index = self.team_id_to_index[matches[home_team_key].iloc[i]]
            away_team_index = self.team_id_to_index[matches[away_team_key].iloc[i]]
            home_goals = matches[home_goals_key].iloc[i]
            away_goals = matches[away_goals_key].iloc[i]
            time = matches[time_key].iloc[i]

            alpha_home = params[home_team_index]
            alpha_away = params[away_team_index]
            beta_home = params[self.nb_teams + home_team_index]
            beta_away = params[self.nb_teams + away_team_index]

            lambda_ = alpha_home * beta_away * gamma
            mu_ = alpha_away * beta_home
            weight = self.weight_fct(time, current_time)

            result += weight * (home_goals * np.log(lambda_) - lambda_ + away_goals * np.log(mu_) - mu_)
            total_weights += weight

            # padding info
            total_weight_per_team[home_team_index] += weight
            total_weight_per_team[away_team_index] += weight

            # TODO: regularization on weight?

        if padding:
            mean_t_weights = np.max(total_weight_per_team)
            for t_weight in total_weight_per_team:
                if t_weight < mean_t_weights * self.padding_intervention_ratio:
                    missing_weight = mean_t_weights * self.padding_intervention_ratio - t_weight
                    result += missing_weight / total_weights * result
                    # we 'add' ('create') few results, as likely as current likelihood

        return float(result)

    def _likelihood_jac(self, matches, params, current_time, padding=True, home_team_key='home_team_id',
                        away_team_key='away_team_id', home_goals_key='home_goals', away_goals_key='away_goals',
                        time_key='date'):
        nb_matches = matches.shape[0]
        gamma = params[2 * self.nb_teams]

        jac = np.zeros_like(params)
        total_weights = 0.
        total_weight_per_team = np.zeros(self.nb_teams)
        for i in range(nb_matches):
            home_team_index = self.team_id_to_index[matches[home_team_key].iloc[i]]
            away_team_index = self.team_id_to_index[matches[away_team_key].iloc[i]]
            home_goals = matches[home_goals_key].iloc[i]
            away_goals = matches[away_goals_key].iloc[i]
            time = matches[time_key].iloc[i]

            alpha_home = params[home_team_index]
            alpha_away = params[away_team_index]
            beta_home = params[self.nb_teams + home_team_index]
            beta_away = params[self.nb_teams + away_team_index]

            weight = self.weight_fct(time, current_time)
            total_weights += weight

            jac[home_team_index] += weight * (home_goals / alpha_home - gamma * beta_away)  # home alpha update
            jac[away_team_index] += weight * (away_goals / alpha_away - beta_home)  # away alpha update

            jac[home_team_index + self.nb_teams] += weight * (away_goals / beta_home - alpha_away)  # home beta
            jac[away_team_index + self.nb_teams] += weight * (home_goals / beta_away - gamma * alpha_home)  # away beta

            jac[self.nb_params-1] += weight * (home_goals / gamma - alpha_home * beta_away)  # gamma update

            # padding info
            total_weight_per_team[home_team_index] += weight
            total_weight_per_team[away_team_index] += weight

            # TODO: regularization on weight ?

        # we 'add' ('create') few results, all with the same chosen score, to 'penalize' team with missing data
        # (most often, team with missing data have been promoted, so they are less good than other teams with data)
        if padding:
            mean_t_weights = np.max(total_weight_per_team)
            for t in range(self.nb_teams):
                t_weight = total_weight_per_team[t]
                if t_weight < mean_t_weights * self.padding_intervention_ratio:
                    # print("-> correction of jac for", self.team_index_to_id[t])
                    missing_weight = mean_t_weights * self.padding_intervention_ratio - t_weight
                    jac[t] += missing_weight * (self.padding_scored/params[t] - (gamma+1.)/2. * 1.)  # alpha update
                    jac[t + self.nb_teams] += missing_weight * (
                                    self.padding_conceded / params[t + self.nb_teams] - (gamma + 1.) / 2. * 1.)  # beta

        return jac

if __name__ == "__main__":
    test_dixon_coles()

