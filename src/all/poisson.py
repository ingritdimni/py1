import math
import numpy as np
from scipy.stats import poisson

SOLVER_DEBUG_MODE = False


# dichotomy implementation, might be improved
def solver(f, x_min, x_max, eps):
    if f(x_min) * f(x_max) > 0:
        if SOLVER_DEBUG_MODE:
            print("no roots between ", x_min, " and ", x_max)
        raise Exception("no solution found in solver")
    else:
        c = (x_min + x_max) / 2.
        while x_max - x_min >= eps:
            c = (x_min + x_max) / 2.
            if f(x_min) * f(c) <= 0:
                x_max = c
            else:
                x_min = c
        return c


class PoissonHelper(object):

    default_min_lambda = 0.05
    default_max_lambda = 8.
    default_precision = 0.01

    def __init__(self):
        pass

    @staticmethod
    def poisson_probability(k_events, lambda_param):
        # naive:   math.exp(-mean) * mean**actual / factorial(actual)
        # iterative, to keep the components from getting too large or small:
        p = math.exp(-lambda_param)
        for i in range(k_events):
            p *= lambda_param
            p /= i + 1
        return p
        # aternative: # return poisson.pmf(k_events, lambda_param)

    @staticmethod
    def match_outcomes_probabilities(lambda_param_1, lambda_param_2, k_max=20):
        table_1 = PoissonHelper.poisson_proba_table(lambda_param_1, k_max)
        table_2 = PoissonHelper.poisson_proba_table(lambda_param_2, k_max)

        p_home_v, p_draw, p_home_d = 0., 0., 0.
        for k in range(k_max + 1):
            p_draw += table_1[k] * table_2[k]
            for l in range(0, k):
                p_home_v += table_1[k] * table_2[l]
            for l in range(k + 1, k_max + 1):
                p_home_d += table_1[k] * table_2[l]

        return p_home_v, p_draw, p_home_d

    @staticmethod
    def play_match(home_param, away_param, seed=None):
        """ creates a match result considering each team param represents its ability to score (poisson distrib)"""
        if seed: np.random.seed(seed)
        home_goals = np.random.poisson(home_param)
        away_goals = np.random.poisson(away_param)
        return home_goals, away_goals

    # approximated (shifted part)
    @staticmethod
    def poisson_proba_table(lambda_param, k_max=10):
        unshifted_prob = [PoissonHelper.poisson_probability(k, lambda_param) for k in range(k_max + 1)]
        total_prob = sum(unshifted_prob)
        shifted_prob = [prob / total_prob for prob in unshifted_prob]
        return shifted_prob

    @staticmethod
    def optimal_poisson_param_given_range(target_probabilities, lambda_1_min, lambda_1_max, precision,
                                          absolute_min_lambda, absolute_max_lambda):

        p1, p2, p3 = target_probabilities

        def g(lambda_param_1):
            f = lambda x: PoissonHelper.match_outcomes_probabilities(lambda_param_1, x)[0] - p1
            lambda_param_2 = solver(f, absolute_min_lambda, absolute_max_lambda, precision)
            return PoissonHelper.match_outcomes_probabilities(lambda_param_1, lambda_param_2)[1] - p2

        # print " \lambda_1_min", lambda_1_min, "     lambda_1_max", lambda_1_max
        lambda_param_1 = solver(g, lambda_1_min, lambda_1_max, precision)
        lambda_param_2 = solver(lambda x: PoissonHelper.match_outcomes_probabilities(lambda_param_1, x)[0] - p1,
                                absolute_min_lambda, absolute_max_lambda, precision)
        return lambda_param_1, lambda_param_2

    # TODO raise specific exception
    @staticmethod
    def implied_param_from_proba(target_probabilities, nb_max_iterations=20, precision=0.000001,
                                 absolute_min_lambda=0.05, absolute_max_lambda=8.):
        lambda_total_possible_range = absolute_max_lambda - absolute_min_lambda
        lambda_sub_range = lambda_total_possible_range / nb_max_iterations
        for i in range(nb_max_iterations):
            try:
                params = PoissonHelper.optimal_poisson_param_given_range(target_probabilities, absolute_min_lambda + i * lambda_sub_range,
                                                                         absolute_min_lambda + (i + 1) * lambda_sub_range, precision,
                                                                         absolute_min_lambda, absolute_max_lambda)
                return params
            except:
                continue
        raise Exception("no solution found !")
