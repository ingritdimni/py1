import numpy as np
from scipy.optimize import minimize


class DixonColes(object):

    def __init__(self, team_universe, weight_fct=None, memory=True):
        """ team universe is an iterable of all involved teams of a given studied universe. Might be IDs or names"""
        self.nb_teams = len(team_universe)
        self.team_index_to_id = dict(zip(range(self.nb_teams), team_universe))
        self.team_id_to_index = dict(zip(team_universe, range(self.nb_teams)))
        if weight_fct is None:
            self.weight_fct = lambda x: 1. # no weights
        else:
            self.weight_fct = weight_fct
        if memory:
            self.param_memory = list()

    def max_likelihood(self, matches, params, home_team_key='home_team_id', away_team_key='away_team_id',
                       home_goals_key='home_goals', away_goals_key='away_goals', time_key='date'):
        # TODO : check matches are all in the past !
        nb_matches = matches.shape[0]
        nb_teams = int((len(params) - 1) / 2)
        result, weights = 0., 0.
        gamma = params[2 * nb_teams]
        # ro = params[2 * nb_teams + 1]
        # zeta = params[2 * nb_teams + 2]
        for i in range(nb_matches):
            home_team_index = self.team_id_to_index[matches[home_team_key][i]]
            away_team_index = self.team_id_to_index[matches[away_team_key][i]]

            home_goals = matches[home_goals_key][i]
            away_goals = matches[away_goals_key][i]
            alpha_home = params[home_team_index]
            alpha_away = params[away_team_key]
            beta_home = params[nb_teams + home_team_index]
            beta_away = params[nb_teams + away_team_index]
            time = matches[time_key][i]

            lambda_ = alpha_home * beta_away * gamma
            mu_ = alpha_away * beta_home
            weight = self.weight_fct(time)

            result += weight * (home_goals * np.log(lambda_) - lambda_ + away_goals * np.log(mu_) - mu_)
            weights += weight

            #TODO: regularization on weight?

        return result

    def max_likelihood_jac(self, matches, params, home_team_key='home_team_id', away_team_key='away_team_id',
                           home_goals_key='home_goals', away_goals_key='away_goals', time_key='date'):
        nb_matches = matches.shape[0]
        gamma = params[2 * self.nb_teams]

        jac = np.zeros(params.size)
        weights = 0.
        for i in range(nb_matches):
            home_team_index = self.team_id_to_index[matches[home_team_key][i]]
            away_team_index = self.team_id_to_index[matches[away_team_key][i]]

            home_goals = matches[home_goals_key][i]
            away_goals = matches[away_goals_key][i]
            alpha_home = params[home_team_index]
            alpha_away = params[away_team_key]
            beta_home = params[self.nb_teams + home_team_index]
            beta_away = params[self.nb_teams + away_team_index]
            time = matches[time_key][i]

            weight = self.weight_fct(time)
            weights += weight

            jac[home_team_index] += weight * (home_goals / alpha_home - gamma * beta_away)  # home alpha update
            jac[away_team_index] += weight * (away_goals / alpha_away - beta_home)  # away alpha update

            jac[home_team_index + self.nb_teams] += weight * (away_goals / beta_home - alpha_away)  # home beta
            jac[away_team_index + self.nb_teams] += weight * (home_goals / beta_away - gamma * alpha_home)  # away beta

            jac[self.nb_teams] += weight * (home_goals / gamma - alpha_home * beta_away)  # gamma update

            # TODO: regularization on weight ?

        return jac
