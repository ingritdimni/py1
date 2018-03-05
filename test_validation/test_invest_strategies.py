import numpy as np
import pandas as pd
from football_betting.create_data import create_noisy_bookmaker_quotes
from football_betting.invest_strategies import ConstantInvestStrategy, ConstantStdDevInvestStrategy, KellyInvestStrategy


def test_invest_strategies():
    matches_probas = pd.DataFrame({'W': [0.4, 0.6, 0.1, 1./3], 'D': [0.2, 0.2, 0.5, 1./3], 'L': [0.4, 0.2, 0.4, 1./3]})
    matches_probas = matches_probas[['W', 'D', 'L']]  # order columns -> very important !
    cst_sigma_invest_strategy = ConstantStdDevInvestStrategy(0.04)
    cst_invest_strategy = ConstantInvestStrategy(1.)
    kelly_invest_strategy = KellyInvestStrategy()
    seed = 2

    for invest_strategy in [cst_invest_strategy, cst_sigma_invest_strategy, kelly_invest_strategy]:
        #print("\n#### testing", invest_strategy.__class__.__name__, "####")
        matches_quotes1 = create_noisy_bookmaker_quotes(matches_probas, std_dev=0.0, fees=0.05, seed=seed)
        invest = invest_strategy.matches_investments(matches_probas, matches_quotes1)
        assert(np.linalg.norm(invest) == 0)  # no investment opportunities ! Bookies are too good (perfect predictions)

        matches_quotes2 = create_noisy_bookmaker_quotes(matches_probas, std_dev=0.02, fees=0.05, seed=seed)
        invest = invest_strategy.matches_investments(matches_probas, matches_quotes2)
        #print(invest)
        assert(np.linalg.norm(invest) > 0)
