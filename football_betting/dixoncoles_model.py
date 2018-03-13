import numpy as np
from scipy.optimize import minimize
from data_preparation import first_data_preparation, create_dict_involved_teams
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
    for i in range(model.nb_teams):
        print(model.team_index_to_id[i], round(opti_params[i], 4), round(opti_params[i+model.nb_teams], 4))
    print('gamma:', round(opti_params[-1], 4))


class DixonColes(object):

    def __init__(self, team_universe, weight_fct=None, memory=True, padding_scored=1., padding_conceded=1.7,
                 padding_intervention_ratio=0.33):
        """ team universe is an iterable of all involved teams of a given studied universe. Might be IDs or names"""
        self.nb_teams = len(team_universe)
        self.nb_params = self.nb_teams * 2 + 1
        self.team_index_to_id = dict(zip(range(self.nb_teams), team_universe))
        self.team_id_to_index = dict(zip(team_universe, range(self.nb_teams)))
        assert(len(self.team_index_to_id) == len(self.team_id_to_index))
        if weight_fct is None:
            self.weight_fct = lambda x: 1.  # no weights
        else:
            self.weight_fct = weight_fct
        if memory:
            self.param_memory = list()
        else:
            self.param_memory = None
        # if not enough data, we 'create' fake data with chosen properties to have more consistent results
        self.padding_scored = padding_scored
        self.padding_conceded = padding_conceded
        self.padding_intervention_ratio = padding_intervention_ratio


    def perform_predictions(self, matches_to_predict, full_matches_history, home_team_key='home_team_id',
                            away_team_key='away_team_id', home_goals_key='home_goals', away_goals_key='away_goals',
                            time_key='date', season_key='season', stage_key='stage'):

        sorted_matches = matches_to_predict.sort([season_key, stage_key, time_key], ascending=[1, 1, 1])
        params_history = list()

        pred_season = sorted_matches[season_key].iloc[0]
        pred_stage = sorted_matches[stage_key].iloc[0]
        pred_time = sorted_matches[time_key].iloc[0]
        relevant_match_history = full_matches_history[full_matches_history[time_key] < pred_time]
        opti_params = self.optimize_parameters(relevant_match_history, home_team_key=home_team_key,
                                               away_team_key=away_team_key, home_goals_key=home_goals_key,
                                               away_goals_key=away_goals_key, time_key=time_key)
        params_history.append([pred_time, opti_params])
        for i, match in sorted_matches.iterrows():
            # parameters calibration
            cur_season = match[season_key]
            cur_stage = match[stage_key]
            cur_time = match[time_key]
            if cur_season != pred_season or cur_stage != pred_stage:
                pred_season, pred_stage, pred_time = cur_season, cur_stage, cur_time
                relevant_match_history = full_matches_history[full_matches_history[time_key] < pred_time]
                # we start optimization by last most relevant params
                opti_params = self.optimize_parameters(relevant_match_history, init_params=opti_params,
                                                       home_team_key=home_team_key, away_team_key=away_team_key,
                                                       home_goals_key=home_goals_key, away_goals_key=away_goals_key,
                                                       time_key=time_key)
                params_history.append([pred_time, opti_params])

            # prediction
            pass

    def optimize_parameters(self, matches, init_params=None, home_team_key='home_team_id', away_team_key='away_team_id',
                            home_goals_key='home_goals', away_goals_key='away_goals', time_key='date'):
        # if self.param_memory and len(self.param_memory):
        #     most_recent_match = matches[time_key].max()
        #     pass  # TODO: load most similar params to accelerate optimization

        # init params
        if init_params is None:
            init_params = np.ones((self.nb_params, 1))

        bounds = ((0.01, 10),) * self.nb_params

        # define local functions involved in optimization
        def constraint_fct(params):  # avg of alphas and betas must be one
            return (np.sum(params[0:self.nb_teams]) - self.nb_teams) ** 2 + \
                   (np.sum(params[self.nb_teams:2*self.nb_teams]) - self.nb_teams)**2

        def constraint_fct_der(params):  # avg of alphas and betas must be one
            jac = np.zeros_like(params)
            alpha_cur = np.sum(params[0:self.nb_teams]) - self.nb_teams
            beta_cur = np.sum(params[self.nb_teams:2*self.nb_teams]) - self.nb_teams
            for i in range(self.nb_teams):
                jac[i] = 2. * params[i] * alpha_cur
            for i in range(self.nb_teams,  2*self.nb_teams):
                jac[i] = 2. * params[i] * beta_cur
            return jac

        def likelihood_m(params):
            return - self._likelihood(matches, params, home_team_key=home_team_key, away_team_key=away_team_key,
                                      home_goals_key=home_goals_key, away_goals_key=away_goals_key, time_key=time_key)

        def likelihood_jac_m(params):
            return - self._likelihood_jac(matches, params, home_team_key=home_team_key, away_team_key=away_team_key,
                                          home_goals_key=home_goals_key, away_goals_key=away_goals_key,
                                          time_key=time_key)

        # other cool methods; TNC or L-BFGS-B
        res = minimize(likelihood_m, init_params, jac=likelihood_jac_m, method='Newton-CG', bounds=bounds,
                       options={'xtol': 10e-3, 'disp': False},
                       constraints=({'type': 'eq', 'fun': constraint_fct, 'jac': constraint_fct_der},))

        # # save params in cache
        # if self.param_memory:
        #     max_t = matches[time_key].max()
        #     self.param_memory.append([max_t, res.x])

        return res.x

    def _likelihood(self, matches, params, padding=True, home_team_key='home_team_id', away_team_key='away_team_id',
                    home_goals_key='home_goals', away_goals_key='away_goals', time_key='date'):
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
            weight = self.weight_fct(time)

            result += weight * (home_goals * np.log(lambda_) - lambda_ + away_goals * np.log(mu_) - mu_)
            total_weights += weight

            # padding info
            total_weight_per_team[home_team_index] += weight
            total_weight_per_team[away_team_index] += weight

            # TODO: regularization on weight?

        if padding:
            mean_t_weights = np.mean(total_weight_per_team)
            for t_weight in total_weight_per_team:
                if t_weight < mean_t_weights * self.padding_intervention_ratio:
                    missing_weight = mean_t_weights * self.padding_intervention_ratio - t_weight
                    result += missing_weight / total_weights * result
                    # we 'add' ('create') few results, as likely as current likelihood

        return float(result)

    def _likelihood_jac(self, matches, params, padding=True, home_team_key='home_team_id', away_team_key='away_team_id',
                        home_goals_key='home_goals', away_goals_key='away_goals', time_key='date'):
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

            weight = self.weight_fct(time)
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
            mean_t_weights = np.mean(total_weight_per_team)
            for t in range(self.nb_teams):
                t_weight = total_weight_per_team[t]
                if t_weight < mean_t_weights * self.padding_intervention_ratio:
                    missing_weight = mean_t_weights * self.padding_intervention_ratio - t_weight
                    jac[t] += missing_weight * (self.padding_scored/1. - (gamma+1.)/2. * 1.)  # alpha update
                    jac[t + self.nb_teams] += missing_weight * (self.padding_conceded/1. - (gamma+1.)/2. * 1.)  # beta

        return jac

if __name__ == "__main__":
    test_dixon_coles()

