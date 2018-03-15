import sqlite3
from time import time
import pandas as pd
import numpy as np

from utils import get_fifa_data, create_feables
from collections import Counter
from fables import simple_fable
from my_utils import get_match_label, match_outcomes_indices, match_outcomes_hot_vectors
from utils_generic import contain_nan

DATA_PATH = "D:/Football_betting/data/soccer/"
DISPLAY_DATA = True
FULL_LOAD = True
REDUCE_DATA_SIZE = False
REMOVE_NA = False

DEFAULT_COUNTRY_SELECTION = ["France", "Spain", "England", "Germany", "Italy"]
# DEFAULT_COUNTRY_SELECTION = ["France", ]


def simple_data_prep(verbose=1, fable_observed_matches=40, padding=False, fable="match_hist",
                     label_format="hot_vectors", remove_nan=True, countries=None):
    data_start = time()
    assert (label_format in ("hot_vectors", "indices", "labels"))
    assert (fable in ("match_hist", "stats"))
    if countries is None:
        countries = DEFAULT_COUNTRY_SELECTION

    if verbose: print(" ... loading data ...")
    player_data, player_stats_data, team_data, match_data = first_data_preparation(DATA_PATH)
    if verbose: print(" ... working with matches from", *countries, '...')
    match_data = match_data.loc[match_data['league_country'].isin(countries)]
    #team_id_to_name, team_name_to_id = create_dict_involved_teams(match_data, team_data)

    if verbose >= 2:
        for name, data in (
                ('player_data', player_data), ('player_stats_data', player_stats_data), ('team_data', team_data),
                ('match_data', match_data)):
            print('\n---', name, data.shape, '---\n', data.head(5), '\n', data.columns.values)

    if verbose: print(" ... creating fables ...")
    match_features = simple_fable(match_data, nb_observed_match=fable_observed_matches, padding=padding,
                                  horizontal_features=False, t_column_name='date', home_team_key='home_team_api_id',
                                  away_team_key='away_team_api_id', home_goals_key='home_team_goal',
                                  away_goals_key='away_team_goal')

    if label_format == "hot_vectors":
        match_labels = match_outcomes_hot_vectors(match_data)
    elif label_format == "indices":
        match_labels = match_outcomes_indices(match_data)
    elif label_format == "labels":
        match_labels = match_data.apply(get_match_label, axis=1)

    if remove_nan:
        shape_init = match_features.shape[0]
        if verbose: print(" ... removing nan ...")
        remove_nan_mask = [not contain_nan(match_features[i]) for i in range(shape_init)]
        match_features = match_features[remove_nan_mask]
        match_labels = match_labels.iloc[remove_nan_mask]
        match_data = match_data.iloc[remove_nan_mask]
        shape = match_features.shape[0]
        if verbose: print('     ', shape_init - shape, 'matches removed;', shape, 'remaining')

    bkm_quotes = pd.DataFrame()
    bkm_quotes['W'], bkm_quotes['D'], bkm_quotes['L'] = match_data['B365H'], match_data['B365D'], match_data['B365A']

    if verbose >= 2:
        print('match_data shape    :', match_data.shape)
        print('match features shape:', match_features.shape)
        print('match labels shape  :', match_labels.shape)
        print('bkm quotes shape    :', bkm_quotes.shape)
    if verbose:
        print(" ... data loaded and prepared in", round(time() - data_start, 2), 'seconds ...\n')

    return match_data, match_features, match_labels, bkm_quotes


def data_prep():
    start = time()

    # load global all, eliminating unwanted features
    player_data, player_stats_data, team_data, match_data = first_data_preparation(DATA_PATH)
    # for n in ('team_long_name', 'team_api_id', 'team_short_name'):
    #     print(n, Counter(team_data[n]))

    df1 = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': ['a', 'b', 'c', 'd', 'e']})
    # df2 = pd.DataFrame({'AA': ['pouet1', 'pouet2', 'pouet3'], 'BB': ['d', 'b', 'a']})
    # print(df1['B'].isin(df2['BB']))
    # print(list(df1['B']))
    # for el in df1.values:
    #     print(el.__class__)
    # team_id_to_name, team_name_to_id = dict(), dict()
    # for index, row in df1.iterrows():
    #     team_id = row['A']
    #     corresponding_name = row['B']
    #     #corresponding_names = df1[df1['A'] == team_id]['B']
    #     #assert (corresponding_names.shape[0] == 1)
    #     #corresponding_name = corresponding_names[0]
    #     team_id_to_name[team_id] = corresponding_name
    #     team_name_to_id[corresponding_name] = team_id
    # assert(len(team_id_to_name) == len(team_name_to_id))
    # print(team_name_to_id)
    # print(team_id_to_name)

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

        # Generating features, exploring the all, and preparing all for model training
        # Generating or retrieving already existant FIFA all
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

            vect_matches, _, _, _ = matches_vectorization(matches, team_data, additional_features=["season", "stage"])
            print(matches.head(5))

            # mask_home = team_data['team_api_id'].isin(matches['home_team_api_id'])
            # mask_away = team_data['team_api_id'].isin(matches['away_team_api_id'])
            # team_universe = team_data[mask_home | mask_away]
            # print(team_universe.head(3))
            # print(team_universe.shape)
            #
            # #team_dummies = pd.get_dummies(matches['home_team_api_id'])
            #
            # prep_dummy_home_matches = matches['home_team_api_id'].astype('category',
            #                                                              categories=team_universe['team_api_id'])
            # dummy_matches_home_teams = pd.get_dummies(prep_dummy_home_matches).astype('int')
            # prep_dummy_away_matches = matches['away_team_api_id'].astype('category',
            #                                                              categories=team_universe['team_api_id'])
            # dummy_matches_away_teams = pd.get_dummies(prep_dummy_away_matches).astype('int')
            #
            # initial_serie = dummy_matches_home_teams.idxmax(axis=1)
            # print(initial_serie.equals(initial_serie))
            # print("####")
            # print(prep_dummy_home_matches)
            # print(prep_dummy_home_matches.shape)
            # print(dummy_matches_away_teams.head(5))
            # print(dummy_matches_away_teams.shape)


            # print(involved_teams.info())
            # print(involved_teams.head(3))
            # print(involved_teams.shape)

    # # Creating features and labels based on all provided
    bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
    bk_cols_selected = ['B365', 'BW']
    # if FULL_LOAD:
    #     feables = create_feables(match_data, fifa_data, bk_cols_selected, get_overall=True)

    # inputs = feables.drop('match_api_id', axis=1)
    #
    # # Exploring the all and creating visualizations
    # labels = inputs.loc[:, 'label']
    # features = inputs.drop('label', axis=1)
    # features.head(5)
    # feature_details = explore_data(features, inputs, path)
    #
    # # Splitting the all into Train, Calibrate, and Test all sets
    # X_train_calibrate, X_test, y_train_calibrate, y_test = train_test_split(features, labels, test_size=0.2,
    #                                                                         random_state=42,
    #                                                                         stratify=labels)
    # X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train_calibrate, y_train_calibrate, test_size=0.3,
    #                                                               random_state=42,
    #                                                               stratify=y_train_calibrate)
    #
    # # Creating cross validation all splits
    # cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
    # cv_sets.get_n_splits(X_train, y_train)


def create_dict_involved_teams(match_data, team_data):
    mask_home = team_data['team_api_id'].isin(match_data['home_team_api_id'])
    mask_away = team_data['team_api_id'].isin(match_data['away_team_api_id'])
    team_universe = team_data[mask_home | mask_away]

    team_id_to_name, team_name_to_id = dict(), dict()
    for index, row in team_universe.iterrows():
        team_id_to_name[row["team_api_id"]] = row["team_long_name"]
        team_name_to_id[row["team_long_name"]] = row["team_api_id"]

    assert(len(team_id_to_name) == len(team_name_to_id))
    return team_id_to_name, team_name_to_id


def matches_vectorization(match_data, team_data, additional_features=None, replace_id_by_names=True):
    mask_home = team_data['team_api_id'].isin(match_data['home_team_api_id'])
    mask_away = team_data['team_api_id'].isin(match_data['away_team_api_id'])
    team_universe = team_data[mask_home | mask_away]
    # print(team_universe.head(3))
    # print(team_universe.shape)
    #print(match_data.head(5))

    team_id_to_name, team_name_to_id = dict(), dict()
    for index, row in team_universe.iterrows():
        team_id_to_name[row["team_api_id"]] = row["team_long_name"]
        team_name_to_id[row["team_long_name"]] = row["team_api_id"]
    assert (len(team_id_to_name) == len(team_name_to_id))
    # print(team_id_to_name)
    # print(team_name_to_id)

    if replace_id_by_names:
        match_data_tmp = match_data.rename(columns={'id': 'id_match'})
        # replace home team id by its name
        match_data_tmp = match_data_tmp.merge(team_data, left_on="home_team_api_id", right_on="team_api_id")
        team_labels_to_drop = list(set(team_data.columns.values) - {'team_long_name'}) + ["home_team_api_id"]
        match_data_tmp = match_data_tmp.drop(labels=team_labels_to_drop, axis=1)
        match_data_tmp = match_data_tmp.rename(columns={'team_long_name': 'home_team'})
        # replace away team id by its name
        match_data_tmp = match_data_tmp.merge(team_data, left_on="away_team_api_id", right_on="team_api_id")
        team_labels_to_drop = list(set(team_data.columns.values) - {'team_long_name'}) + ["away_team_api_id"]
        match_data_tmp = match_data_tmp.drop(labels=team_labels_to_drop, axis=1)
        match_data_tmp = match_data_tmp.rename(columns={'team_long_name': 'away_team'})
        match_data_tmp = match_data_tmp.sort_values('id_match')
        match_data_tmp = match_data_tmp.reset_index()
    else:
        match_data_tmp = match_data.reset_index()
    #print(match_data_tmp.head(5))

    kept_features = ["home_team_goal", "away_team_goal"]
    if replace_id_by_names:
        kept_features += ['home_team', 'away_team']
        transformed_featured = ['home_team', 'away_team']
    else:
        kept_features += ['home_team_api_id', 'away_team_api_id']
        transformed_featured = ['home_team_api_id', 'away_team_api_id']
    if additional_features: kept_features += additional_features
    removed_features = list(set(match_data_tmp.columns.values) - set(kept_features))
    vect_matches = match_data_tmp.drop(labels=removed_features, axis=1)
    #print(vect_matches.head(5))
    vect_matches = pd.get_dummies(vect_matches, columns=transformed_featured)
    #print(vect_matches.head(5))

    return vect_matches, team_universe, team_id_to_name, team_name_to_id


def first_data_preparation(data_path=DATA_PATH):

    # Fetching all
    # Connecting to database
    database = data_path + 'database.sqlite'
    conn = sqlite3.connect(database)

    # Fetching required all tables
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

    # remove matchs with missing all TODO: ensure we want to remove matches with results but no bkm all
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
    # extended_match_features = ['shoton', 'shotoff', 'possession']  # there are some error in the all, to be solved later on
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

    # Reduce match all to fulfill run time requirements
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
    # data_prep()
    simple_data_prep()
