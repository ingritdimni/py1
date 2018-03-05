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
                                                    'expected_return': expected_return}, ignore_index=True)

        return df_investments


class ConstantInvestStrategy(GenericInvestStrategy):
    def __init__(self, constant_investment, choose_outcome_fct=None):
        """ initialize ConstantInvestStrategy """
        self.constant_investment = constant_investment
        if choose_outcome_fct:
            self.choose_outcome_fct = choose_outcome_fct
        else:
            # default choose_outcome_fct is gated_match_best_return_outcome with investment if return_expectation > 1 %
            self.choose_outcome_fct = lambda x, y: gated_match_best_return_outcome(x, y, 0.01)

    def match_investment(self, match_probas, match_quote):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is (investment_index, investment, expected_return)"""
        p, b = list(match_probas), list(match_quote)
        outcome_i, expected_return = self.choose_outcome_fct(p, b)
        if outcome_i is None:
            return 0., 0., 0.  # we invest 0 on win !
        return outcome_i, self.constant_investment, self.constant_investment * expected_return


class ConstantStdDevInvestStrategy(GenericInvestStrategy):
    def __init__(self, sigma, choose_outcome_fct=None):
        """ initialize ConstantStdDevInvestSrategy """
        self._sigma = sigma
        if choose_outcome_fct:
            self.choose_outcome_fct = choose_outcome_fct
        else:
            # default choose_outcome_fct is gated_match_best_return_outcome with investment if return_expectation > 1 %
            self.choose_outcome_fct = lambda x, y: gated_match_best_return_outcome(x, y, 0.01)

    def match_investment(self, match_probas, match_quote):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is (investment_index, proportion_of_wealth_to_invest, proportion_of_wealth_expected_return)"""
        p, b = list(match_probas), list(match_quote)
        outcome_i, expected_return = self.choose_outcome_fct(p, b)
        if outcome_i is None:
            return 0., 0., 0.  # we invest 0 on win !
        invest_sigma = b[outcome_i] * np.sqrt(p[outcome_i] * (1. - p[outcome_i]))
        return outcome_i, self._sigma / invest_sigma, self._sigma / invest_sigma * expected_return


class KellyInvestStrategy(GenericInvestStrategy):
    def __init__(self, choose_outcome_fct=None):
        """ initialize ConstantStdDevInvestSrategy """
        if choose_outcome_fct:
            self.choose_outcome_fct = choose_outcome_fct
        else:
            # default choose_outcome_fct is gated_match_best_return_outcome with investment if return_expectation > 1 %
            self.choose_outcome_fct = lambda x, y: gated_match_best_return_outcome(x, y, 0.01)

    def match_investment(self, match_probas, match_quote):
        """ return investment to make, in proportion of current wealth, to get a theoretical constant risk sigma.
        Output is (investment_index, proportion_of_wealth_to_invest, proportion_of_wealth_expected_return)"""
        p, b = list(match_probas), list(match_quote)
        outcome_i, expected_return = self.choose_outcome_fct(p, b)
        if outcome_i is None:
            return 0., 0., 0.  # we invest 0 on win !
        investment = p[outcome_i] - (1. - p[outcome_i]) / (b[outcome_i] - 1.)
        return outcome_i, investment, investment * expected_return


if __name__ == "__main__":
    test_invest_strategies()
