import sqlite3
from time import time
import pandas as pd
import numpy as np

from utils import get_fifa_data, create_feables
from collections import Counter

DATA_PATH = "D:/Football_betting/data/soccer/"
DISPLAY_DATA = True
FULL_LOAD = True
REDUCE_DATA_SIZE = False
REMOVE_NA = False

COUNTRY_SELECTION = 'France'


def main():
    start = time()

    # load global data, eliminating unwanted features
    player_data, player_stats_data, team_data, match_data = first_data_preparation(DATA_PATH)
    # for n in ('team_long_name', 'team_api_id', 'team_short_name'):
    #     print(n, Counter(team_data[n]))

    df1 = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']})
    df2 = pd.DataFrame({'AA': ['pouet1', 'pouet2', 'pouet3'], 'BB': ['d', 'b', 'a']})
    print(df1['B'].isin(df2['BB']))
    print(list(df1['B']))

    # all_countries =['England', 'France', 'Spain', 'Italy', 'Germany', 'Netherlands', 'Portugal', 'Poland', 'Scotland',
    #                 'Belgium', 'Switzerland']
    studied_countries = ['France']
    for country in studied_countries:
        print('\n### Working With', country, 'only ###\n')
        matches = match_data.loc[match_data['league_country'] == country]
        # to run faster, work on reduced match sample ? TODO: improve selection
        if REDUCE_DATA_SIZE: matches = matches.tail(200)

        # advanced header display
        if DISPLAY_DATA:
            for name, data in (
            ('player_data', player_data), ('player_stats_data', player_stats_data), ('team_data', team_data),
            ('match_data', matches)):
                print('\n---', name, data.shape, '---\n', data.head(5), '\n', data.columns.values)


        # dummies = pd.get_dummies(match_data['league_country'])
        # print(dummies)

        # Generating features, exploring the data, and preparing data for model training
        # Generating or retrieving already existant FIFA data
        if FULL_LOAD:
            # TODO: fill missing values with likely ones (mean of ctry ? third decile ?)
            #fifa_data = get_fifa_data(matches, player_stats_data, data_exists=False)
            # if DISPLAY_DATA:
            #     print("----------")
            #     print(fifa_data.info())
            #     print("----------")
            #     print(fifa_data.head(3))
            #     print("----------")
            #     print(fifa_data.shape)

            mask_home = team_data['team_api_id'].isin(matches['home_team_api_id'])
            mask_away = team_data['team_api_id'].isin(matches['away_team_api_id'])
            involved_teams = team_data[mask_home | mask_away]
            print(involved_teams.head(3))
            print(involved_teams.shape)

            #team_dummies = pd.get_dummies(matches['home_team_api_id'])

            prep_dummy = matches['home_team_api_id'].astype('category', categories=involved_teams['team_api_id'])
            dummy_matches_home_teams = pd.get_dummies(prep_dummy).astype('int')

            initial_serie = dummy_matches_home_teams.idxmax(axis=1)
            print(initial_serie.equals(initial_serie))
            print("####")
            print(prep_dummy)
            print(prep_dummy.shape)
            # print(initial_serie)
            # print("####")
            # print(matches['home_team_api_id'])

            my_team_long_name = "FC Nantes"  # team_api_id = 9830
            my_team_api_id = 8583  # "AJ Auxerre"
            key, input, other_key = "team_api_id", my_team_api_id, "team_long_name"
            key2, input2, other_key2 = "team_long_name", my_team_long_name, "team_api_id"

            # mapping use team universe, which is "involved_teams"
            def my_mapping(key, input, other_key):
                res = involved_teams[involved_teams[key] == input][other_key]
                assert(res.shape[0] == 1)
                return res.values[0]

            assert(my_mapping(key2, input2, other_key2) == 9830)
            assert(my_mapping(key, input, other_key) == "AJ Auxerre")
            # print(my_mapping(key2, input2, other_key2))
            # print(my_mapping(key, input, other_key))




            # TODO: fct hotrepresentation -> team_id representation
            # TODO: fct team_id serie -> hotrepresentation


            # print(involved_teams.info())
            # print(involved_teams.head(3))
            # print(involved_teams.shape)

    # # Creating features and labels based on data provided
    bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
    bk_cols_selected = ['B365', 'BW']
    # if FULL_LOAD:
    #     feables = create_feables(match_data, fifa_data, bk_cols_selected, get_overall=True)

    # inputs = feables.drop('match_api_id', axis=1)
    #
    # # Exploring the data and creating visualizations
    # labels = inputs.loc[:, 'label']
    # features = inputs.drop('label', axis=1)
    # features.head(5)
    # feature_details = explore_data(features, inputs, path)
    #
    # # Splitting the data into Train, Calibrate, and Test data sets
    # X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(features, labels, test_size=0.2,
    #                                                                         random_state=42,
    #                                                                         stratify=labels)
    # X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train_calibrate, y_train_calibrate, test_size=0.3,
    #                                                               random_state=42,
    #                                                               stratify=y_train_calibrate)
    #
    # # Creating cross validation data splits
    # cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
    # cv_sets.get_n_splits(X_train, y_train)


def first_data_preparation(data_path):

    # Fetching data
    # Connecting to database
    database = data_path + 'database.sqlite'
    conn = sqlite3.connect(database)

    # Fetching required data tables
    player_data = pd.read_sql("SELECT * FROM Player;", conn)
    player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
    team_data = pd.read_sql("SELECT * FROM Team;", conn)
    match_data = pd.read_sql("SELECT * FROM Match;", conn)
    countries_data = pd.read_sql_query("SELECT * FROM Country;", conn)

    # remove unused features
    player_data, player_stats_data, team_data, match_data = drop_useless_features(player_data, player_stats_data,
                                                                                  team_data, match_data, countries_data)

    # if DISPLAY_DATA:
    #     print('Counter before na cleaning:\n', Counter(match_data['league_country']))

    # remove matchs with missing data TODO: ensure we want to remove matches with results but no bkm data
    if REMOVE_NA:
        player_data, player_stats_data, team_data, match_data = drop_na_data(player_data, player_stats_data, team_data,
                                                                             match_data)

    return player_data, player_stats_data, team_data, match_data


def drop_useless_features(player_data, player_stats_data, team_data, match_data, countries_data,
                          booky_preselection=None):
    """" remove unwanted features (arbitrary chosen)"""

    if not booky_preselection:  # those booky quotes will be kept
        booky_preselection = ['B365', 'BW']

    player_features_to_keep = ['id', 'player_api_id', 'player_name', 'player_fifa_api_id']
    player_features_to_drop = list(set(player_data.columns.values) - set(player_features_to_keep))
    player_data.drop(player_features_to_drop, axis=1, inplace=True)

    player_stats_features_to_keep = ['id', 'player_fifa_api_id', 'player_api_id', 'date', 'overall_rating', 'potential']
    player_stats_features_to_drop = list(set(player_stats_data.columns.values) - set(player_stats_features_to_keep))
    player_stats_data.drop(player_stats_features_to_drop, axis=1, inplace=True)

    team_features_to_keep = ['id', 'team_api_id', 'team_fifa_api_id', 'team_long_name', 'team_short_name']
    team_features_to_drop = list(set(team_data.columns.values) - set(team_features_to_keep))
    team_data.drop(team_features_to_drop, axis=1, inplace=True)

    basic_match_features = ["id", "country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
                            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
                            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
                            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
                            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
                            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    extended_match_features = list()
    # extended_match_features = ['shoton', 'shotoff', 'possession']  # there are some error in the data, to be solved later on
    additional_match_features = [bkm + label for bkm in booky_preselection for label in ('H', 'D', 'A')]
    match_features_to_keep = basic_match_features + extended_match_features + additional_match_features
    match_features_to_drop = list(set(match_data.columns.values) - set(match_features_to_keep))
    match_data.drop(match_features_to_drop, axis=1, inplace=True)

    # last but not least, we replace countries id by their names; more readable !
    countries_data = countries_data.rename(columns={'id': 'id_c'})
    match_data = match_data.merge(countries_data, left_on="league_id", right_on="id_c")
    match_data = match_data.drop(labels=["id_c", "league_id", "country_id"], axis=1)
    match_data = match_data.rename(columns={'name': 'league_country'})

    #TODO: cast types ? using #match_data['home_player_1'] = match_data['home_player_1'].astype(int) for ex

    return player_data, player_stats_data, team_data, match_data


def drop_na_data(player_data, player_stats_data, team_data, match_data):

    # Reduce match data to fulfill run time requirements
    # rows = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
    #         "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
    #         "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
    #         "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
    #         "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
    #         "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
    # match_data.dropna(subset=rows, inplace=True)  # reduce nb of matches to 21374
    match_data = match_data.dropna(axis=0).reset_index(drop=True)
    return player_data, player_stats_data, team_data, match_data

if __name__ == '__main__':
    main()
