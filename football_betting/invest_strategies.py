import numpy as np
import pandas as pd
from create_data import create_noisy_bookmaker_quotes


def match_outcomes_expected_returns(match_probas, match_quote, is_input_a_list=True):
    """ given a match distribution (probs of [W, D, L]) and corresponding bookmaker quotes, return expected returns """
    assert(len(match_probas) == len(match_quote) == 3)
    if is_input_a_list:
        return [match_probas[i] * (match_quote[i] - 1.) - (1 - match_probas[i]) for i in range(3)]
    return [match_probas.iloc[:, i] * (match_quote.iloc[:, i] - 1.) - (1 - match_probas.iloc[:, i]) for i in range(3)]


def match_best_return_outcome(match_probas, match_quote):
    """ given a match distribution (probs of [W, D, L]) and corresponding bookmaker quotes,
    return index corresponding to best expected returns and corresponding best expected return"""
    expected_returns = match_outcomes_expected_returns(match_probas, match_quote)
    best_return = max(expected_returns)
    return expected_returns.index(best_return), best_return


def gated_match_best_return_outcome(match_probas, match_quote, gate):
    """ given a match distribution (probs of [W, D, L]) and corresponding bookmaker quotes,
    return index corresponding to best expected returns if > gate and best expected return, else return None """
    expected_returns = match_outcomes_expected_returns(match_probas, match_quote)
    best_return = max(expected_returns)
    if best_return >= gate:
        return expected_returns.index(max(expected_returns)), best_return
    return None, 0.


class GenericInvestStrategy(object):
    def __init__(self):
        pass

    def match_investment(self, match_probas, match_quote):
        """ abstract method to get match investment on a given match considering inputs"""
        raise NotImplementedError("match_investment should be overwritten in child class")

    def matches_investments(self, matches_probas, matches_quotes):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is a dataframe with columns ("investment_index", "investment", "expected_return")
        where investment and expected_return are proportions of total wealth """
        nb_matches = matches_probas.shape[0]
        assert(nb_matches == matches_quotes.shape[0])

        is_probas_pd_type, is_quotes_pd_type = False, False
        if isinstance(matches_probas, pd.DataFrame) or isinstance(matches_probas, pd.Series):
            is_probas_pd_type = True
        if isinstance(matches_quotes, pd.DataFrame) or isinstance(matches_quotes, pd.Series):
            is_quotes_pd_type = True

        df_investments = pd.DataFrame(columns=['investment_index', 'investment', 'expected_return'])
        for i in range(nb_matches):
            p = list(matches_probas.iloc[i, :]) if is_probas_pd_type else list(matches_probas[i, :])
            q = list(matches_quotes.iloc[i, :]) if is_quotes_pd_type else list(matches_quotes[i, :])

            outcome_i, investment, expected_return = self.match_investment(p, q)
            df_investments = df_investments.append({'investment_index': outcome_i, 'investment': investment,
                                                    'expected_return': expected_return, 'quote': q[outcome_i]},
                                                   ignore_index=True)

        return df_investments

    @staticmethod
    def matches_gains(actual_results, df_investments):
        # the below does not work on Y_val, i have no clue why
        # df_investments['results'] = actual_results.idxmax(axis=1)
        # df_investments['result'] = actual_results.apply(lambda x: [x['W'], x['D'], x['L']].index(1), axis=1)
        df_investments['result_index'] = pd.Series(
            [[actual_results['W'].iloc[i], actual_results['D'].iloc[i], actual_results['L'].iloc[i]].index(1) for i in
             range(actual_results.shape[0])])
        df_investments['gain'] = df_investments.apply(
            lambda x: x['investment'] * (x['quote'] - 1.) if x['result_index'] == x['investment_index'] else -x['investment'],
            axis=1)

        return df_investments

    @staticmethod
    def precompute_wealth_affine(matches_gains, init_wealth=100, time_feature=None):
        """ computes current wealth after each bet.
        Assumes matches gain are already in money units, i.e. not relative to current wealth """
        wealth, invested_amounts, gain_amounts, exp_gain_amounts = list(), list(), list(), list()
        cur_wealth = init_wealth
        for i in range(matches_gains.shape[0]):
            gain_amount = matches_gains['gain'].iloc[i]
            cur_wealth = cur_wealth + gain_amount
            wealth.append(cur_wealth)
            gain_amounts.append(gain_amount)
            invested_amounts.append(matches_gains['investment'].iloc[i])
            exp_gain_amounts.append(matches_gains['expected_return'].iloc[i])
        wealth = pd.Series(wealth, name='wealth')
        invested_amounts = pd.Series(invested_amounts, name='invested_amount')
        gain_amounts = pd.Series(gain_amounts, name='gain_amount')
        exp_return_amounts = pd.Series(exp_gain_amounts, name='exp_gain_amounts')
        return invested_amounts, exp_return_amounts, gain_amounts, wealth

    @staticmethod
    def precompute_wealth_proportional(matches_gains, init_wealth=100, time_feature=None):
        """ computes current wealth after each bet.
        Assumes matches gain are relative to current wealth.
        indicates time_key to give time columns name if this method has to handle several investments at a given time
        (i.e. with a shared current wealth)"""
        # wealth = pd.DataFrame(columns=['wealth'])
        wealth, invested_amounts, gain_amounts, exp_gain_amounts = list(), list(), list(), list()
        cur_wealth = init_wealth
        ref_wealth, last_wealth = init_wealth, init_wealth
        if time_feature is not None:
            assert(time_feature.shape[0] == matches_gains.shape[0])
            if matches_gains.shape[0] > 0:
                last_t = time_feature.iloc[0]
        for i in range(matches_gains.shape[0]):
            if time_feature is not None:  # handle several investments to be made on the same time
                cur_t = time_feature.iloc[0]
                assert(last_t <= cur_t)
                if last_t < cur_t:  # if time advanced, then reference wealth is updated
                    ref_wealth = last_wealth
                invested_amount = matches_gains['investment'].iloc[i] * ref_wealth
                gain_amount = matches_gains['gain'].iloc[i] * ref_wealth
                exp_gain_amount = matches_gains['expected_return'].iloc[i] * ref_wealth
                cur_wealth += gain_amount
                last_wealth = cur_wealth
                last_t = cur_t
            else:
                invested_amount = matches_gains['investment'].iloc[i] * cur_wealth
                gain_amount = matches_gains['gain'].iloc[i] * cur_wealth
                exp_gain_amount = matches_gains['expected_return'].iloc[i] * ref_wealth
                cur_wealth += gain_amount
            invested_amounts.append(invested_amount)
            gain_amounts.append(gain_amount)
            wealth.append(cur_wealth)
            exp_gain_amounts.append(exp_gain_amount)
        wealth = pd.Series(wealth, name='wealth')
        invested_amounts = pd.Series(invested_amounts, name='invested_amount')
        gain_amounts = pd.Series(gain_amounts, name='gain_amount')
        exp_gain_amounts = pd.Series(exp_gain_amounts, name='exp_return_amount')
        return invested_amounts, exp_gain_amounts, gain_amounts, wealth

    @staticmethod
    def precompute_wealth(matches_gains, init_wealth=100, time_feature=None):
        """ abstract method to get wealth computations """
        raise NotImplementedError('wealth_computation should be overwritten in child class')

    def apply_invest_strategy(self, matches_probas, matches_quotes, actual_results, init_wealth=100, time_feature=None):
        df_investments = self.matches_investments(matches_probas, matches_quotes)
        df_gains = self.matches_gains(actual_results, df_investments)
        invested_amounts, exp_gain_amount, gain_amounts, wealth = self.precompute_wealth(df_gains, init_wealth, time_feature)
        df = pd.DataFrame()
        df['invested_amounts'], df['exp_gain_amounts'] = invested_amounts, exp_gain_amount
        df['gain_amounts'], df['wealth'] = gain_amounts, wealth
        return df


class ConstantAmountInvestStrategy(GenericInvestStrategy):
    """ Systematically invest constant_amount units on each bet, whatever current wealth is """

    def __init__(self, constant_amount, choose_outcome_fct=None, min_return=0.01):
        """ initialize ConstantInvestStrategy """
        self.constant_amount = constant_amount
        if choose_outcome_fct:
            self.choose_outcome_fct = choose_outcome_fct
        else:
            # default choose_outcome_fct is gated_match_best_return_outcome with investment if return_expectation > 1 %
            self.choose_outcome_fct = lambda x, y: gated_match_best_return_outcome(x, y, min_return)

    def match_investment(self, match_probas, match_quote):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is (investment_index, investment, expected_return)"""
        p, b = list(match_probas), list(match_quote)
        outcome_i, expected_return = self.choose_outcome_fct(p, b)
        if outcome_i is None:
            return 0, 0., 0.  # we invest 0 on win !
        return outcome_i, self.constant_amount, self.constant_amount * expected_return

    @staticmethod
    def precompute_wealth(matches_gains, init_wealth=100, time_feature=None):
        return GenericInvestStrategy.precompute_wealth_affine(matches_gains, init_wealth=init_wealth,
                                                              time_feature=time_feature)


class ConstantPercentInvestStrategy(GenericInvestStrategy):
    """ Systematically invest constant_percent of wealth on each bet """

    def __init__(self, constant_percent, choose_outcome_fct=None, min_return=0.01):
        """ initialize ConstantInvestStrategy """
        self.constant_percent = constant_percent
        if choose_outcome_fct:
            self.choose_outcome_fct = choose_outcome_fct
        else:
            # default choose_outcome_fct is gated_match_best_return_outcome with investment if return_expectation > 1 %
            self.choose_outcome_fct = lambda x, y: gated_match_best_return_outcome(x, y, min_return)

    def match_investment(self, match_probas, match_quote):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is (investment_index, investment, expected_return)"""
        p, b = list(match_probas), list(match_quote)
        outcome_i, expected_return = self.choose_outcome_fct(p, b)
        if outcome_i is None:
            return 0, 0., 0.  # we invest 0 on win !
        return outcome_i, self.constant_percent, self.constant_percent * expected_return

    @staticmethod
    def precompute_wealth(matches_gains, init_wealth=100, time_feature=None):
        return GenericInvestStrategy.precompute_wealth_proportional(matches_gains, init_wealth=init_wealth,
                                                                    time_feature=time_feature)


class ConstantStdDevInvestStrategy(GenericInvestStrategy):
    def __init__(self, sigma, choose_outcome_fct=None, min_return=0.01):
        """ initialize ConstantStdDevInvestSrategy """
        self._sigma = sigma
        if choose_outcome_fct:
            self.choose_outcome_fct = choose_outcome_fct
        else:
            # default choose_outcome_fct is gated_match_best_return_outcome with investment if return_expectation > 1 %
            self.choose_outcome_fct = lambda x, y: gated_match_best_return_outcome(x, y, min_return)

    def match_investment(self, match_probas, match_quote):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is (investment_index, proportion_of_wealth_to_invest, proportion_of_wealth_expected_return)"""
        p, b = list(match_probas), list(match_quote)
        outcome_i, expected_return = self.choose_outcome_fct(p, b)
        if outcome_i is None:
            return 0, 0., 0.  # we invest 0 on win !
        invest_sigma = b[outcome_i] * np.sqrt(p[outcome_i] * (1. - p[outcome_i]))
        return outcome_i, self._sigma / invest_sigma, self._sigma / invest_sigma * expected_return

    @staticmethod
    def precompute_wealth(matches_gains, init_wealth=100, time_feature=None):
        return GenericInvestStrategy.precompute_wealth_proportional(matches_gains, init_wealth=init_wealth,
                                                                    time_feature=time_feature)


class KellyInvestStrategy(GenericInvestStrategy):
    def __init__(self, choose_outcome_fct=None, min_return=0.01):
        """ initialize ConstantStdDevInvestSrategy """
        if choose_outcome_fct:
            self.choose_outcome_fct = choose_outcome_fct
        else:
            # default choose_outcome_fct is gated_match_best_return_outcome with investment if return_expectation > 1 %
            self.choose_outcome_fct = lambda x, y: gated_match_best_return_outcome(x, y, min_return)

    def match_investment(self, match_probas, match_quote):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is (investment_index, proportion_of_wealth_to_invest, proportion_of_wealth_expected_return)"""
        p, b = list(match_probas), list(match_quote)
        outcome_i, expected_return = self.choose_outcome_fct(p, b)
        if outcome_i is None:
            return 0, 0., 0.  # we invest 0 on win !
        investment = p[outcome_i] - (1. - p[outcome_i]) / (b[outcome_i] - 1.)
        return outcome_i, investment, investment * expected_return

    @staticmethod
    def precompute_wealth(matches_gains, init_wealth=100, time_feature=None):
        return GenericInvestStrategy.precompute_wealth_proportional(matches_gains, init_wealth=init_wealth,
                                                                    time_feature=time_feature)

if __name__ == "__main__":
    pass
