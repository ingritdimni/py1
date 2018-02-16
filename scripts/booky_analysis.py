import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

sns.set(style='whitegrid', context='notebook', palette='deep')

np.random.seed(2)

DATA_PATH = "D:/Football_betting/data/soccer/"


def main():

    con = sqlite3.connect(DATA_PATH + "database.sqlite")

    matches = pd.read_sql_query("SELECT * FROM Match;", con)
    teams = pd.read_sql_query("SELECT * FROM Team;", con)
    # leagues = pd.read_sql_query("SELECT * FROM League;",con)
    countries = pd.read_sql_query("SELECT * FROM Country;", con)

    print(matches.head())

    matches = matches.drop(labels=['home_player_X1', 'home_player_X2', 'home_player_X3',
                                   'home_player_X4', 'home_player_X5', 'home_player_X6',
                                   'home_player_X7', 'home_player_X8', 'home_player_X9',
                                   'home_player_X10', 'home_player_X11',
                                   'home_player_Y1', 'home_player_Y2', 'home_player_Y3',
                                   'home_player_Y4', 'home_player_Y5', 'home_player_Y6',
                                   'home_player_Y7', 'home_player_Y8', 'home_player_Y9',
                                   'home_player_Y10', 'home_player_Y11', 'home_player_1',
                                   'home_player_2', 'home_player_3', 'home_player_4',
                                   'home_player_5', 'home_player_6', 'home_player_7',
                                   'home_player_8', 'home_player_9', 'home_player_10',
                                   'home_player_11', 'away_player_X1', 'away_player_X2',
                                   'away_player_X3', 'away_player_X4',
                                   'away_player_X5', 'away_player_X6', 'away_player_X7',
                                   'away_player_X8', 'away_player_X9', 'away_player_X10',
                                   'away_player_X11', 'away_player_Y2', 'away_player_Y3',
                                   'away_player_Y4', 'away_player_Y5', 'away_player_Y6',
                                   'away_player_Y7', 'away_player_Y1',
                                   'away_player_Y8', 'away_player_Y9', 'away_player_Y10',
                                   'away_player_Y11', 'away_player_1',
                                   'away_player_2', 'away_player_3', 'away_player_4',
                                   'away_player_5', 'away_player_6', 'away_player_7',
                                   'away_player_8', 'away_player_9', 'away_player_10',
                                   'away_player_11'], axis=1)

    matches = matches.drop(labels=['goal', 'shoton', 'shotoff', 'foulcommit', 'card',
                                   'cross', 'corner', 'possession'], axis=1)

    matches = matches.drop(labels=['GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA', 'SJH', 'SJD', 'SJA',
                                   'PSH', 'PSD', 'PSA'], axis=1)

    matches = matches.dropna(axis=0).reset_index(drop=True)

    print(matches.shape)  # (22432, 29)

    print(matches.isnull().sum())  # The dataset is ready, no missing value remains.

    # on the below, we search for the most predictable league

    matches = matches.merge(countries, left_on="league_id", right_on="id")

    matches = matches.drop(labels=["id_y", "league_id", "country_id"], axis=1)

    matches = matches.rename(columns={'name': 'league_country'})

    matches["result"] = (matches["home_team_goal"] - matches["away_team_goal"]).map(
        lambda s: 0 if s > 0 else 1 if s == 0 else 2)  # (0 = home team victory, 1 = Draw, 2 = Away team victory)

    bookies = ('B365', 'BW', 'IW', 'LB', 'WH', 'VC')
    match_issues = ('H', 'D', 'A')

    probabilities = {}
    for bkm in bookies:
        #print([bkm + m_issue for m_issue in match_issues])
        matches[bkm] = np.argmin(matches[[bkm + m_issue for m_issue in match_issues]].values, axis=1)

        probabilities[bkm] = np.reciprocal(matches[[bkm + m_issue for m_issue in match_issues]].values)
        row_sums = probabilities[bkm].sum(axis=1).reshape((-1, 1))
        probabilities[bkm] = np.divide(probabilities[bkm], row_sums)

    # who is best at predicting match results ?
    for bkm in bookies:
        print('avg victory prediction', bkm, accuracy_score(matches["result"], matches[bkm]))
        # the below score has to be manually validated by runninng it manually on 2-3 matchs only
        #print('log likelyhood score', bkm, log_loss(matches["result"], probabilities[bkm]))

    # Compute accuracy in each group in the groupby pandas objects
    def acc_group(y_true_desc, y_pred_desc):
        def inner(group):
            return accuracy_score(group[y_true_desc], group[y_pred_desc])

        inner.__name__ = 'acc_group'
        return inner

    matches.groupby("league_country").apply(acc_group("result", "B365"))

    league_seasons_accuracies = matches.groupby(("league_country", "season")).apply(acc_group("result", "B365"))
    league_seasons_accuracies = league_seasons_accuracies.reset_index()
    league_seasons_accuracies = league_seasons_accuracies.rename(columns={0: 'accuracy'})
    selected_countries = ["France", "Spain", "England", "Germany", "Italy"]

    Five_leagues = league_seasons_accuracies[league_seasons_accuracies['league_country'].isin(selected_countries)]

    g = sns.factorplot(x="season", y="accuracy", hue="league_country", data=Five_leagues, size=6, aspect=1.5)
    g.set_xticklabels(rotation=45)
    plt.suptitle('Bet 365 accuracy for the 5 biggest soccer leagues')
    plt.show()


if __name__ == '__main__':
    main()


