import json
import pandas as pd
import numpy as np
import csv
import sys
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, recall_score, precision_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor 
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

# Kento Croft
# Mark 19.82/20
# useage: python3 <renenue_rating_predictor.py> <training.csv> <validation.csv>
    # This program is specifically made to handle the csv formats provided
# output: MSE_correlation.csv and revenue_predictions.csv

# ===============================================Helper functions=============================================== 

def to_json(raw_string):
    return json.loads(raw_string)

def production_company_names(production_companies):
    return [production_company['name'] for production_company in production_companies]

def cast_names(casts):
    return [cast['name'] for cast in casts]

def find_directors(crews):
    dictories = []
    for crew in crews:
        if crew['job'] == 'Director':
            dictories.append(crew['name'])
    return dictories 

def find_important_jobs(crews):
    jobs = []
    for crew in crews:
        if crew['job'] == 'Director':
            jobs.append(crew['name'])
        elif crew['job'] == 'Writer':
            jobs.append(crew['name'])
        elif crew['job'] == 'Executive Producer':
            jobs.append(crew['name'])
    return jobs  

def has_webpage(col):
    return 0 if type(col) == float else 1

def month_day(col):
    date = col.split('-')
    if len(date) == 3:
        return int(date[1]+date[2])

def find_keywords(words):
    key_words = []
    for keywords in words:
        key_words.append(keywords['name'])
    return key_words

def find_genres(genres):
    genres_list = []
    for genre in genres:
        genres_list.append(genre['name'])
    return genres_list


# Load in training and validation data sets into seperate dataframes 
training_df =  pd.read_csv(sys.argv[1]) # This will be used for training (and hyper-parameter finding)
testing_df = pd.read_csv(sys.argv[2]) # This will only be used for testing

# ===============================================PART 1 regression=============================================== 

# model randomforestregression 

# Features extracted were:
    # Top X most frequently appearing cast members
    # Top X most frequently appearing crew members (considering only directors, writers and executive producers)
    # Top X most frequently appearing genres 
    # Top X most frequently appearing production_companies 
    # Homepage 1 or 0 
    # cast_size

# note: For all the 'top X' features, each member becomes their own feature (column) one hot encoded
# meaning for example, top X most frequent cast members, might become 100 + new columns, each named with the individuals name, one hot encoded

# ======== preprocessing ========
X = training_df.loc[:, ['cast', 'crew', 'genres', 'homepage', 'production_companies', 'revenue']] # get relavent columns only
X.loc[:,['cast', 'crew', 'production_companies', 'genres']] = X.loc[:,['cast', 'crew', 'production_companies', 'genres']].applymap(to_json) # convert string to json
X.production_companies = X.production_companies.apply(production_company_names) # converts messy json list to just list of company names  
X.cast = X.cast.apply(cast_names) # converts messy json list to just list of cast names
X['cast_size'] = X.cast.apply(lambda cast: len(cast)) # makes new column with the size of each cast
X.crew = X.crew.apply(find_important_jobs) # converts messy json list to just list of crew names
X.genres = X.genres.apply(find_genres) # converts messy json list to just list of grenes
production_company_freq = X.explode('production_companies').production_companies.value_counts() # Create an ordered series of the frequencies of companies 
Most_common_actors = X.explode('cast').cast.value_counts() # Create an ordered series of the frequencies of names 
Most_common_important_jobs = X.explode('crew').crew.value_counts() # Create an ordered series of the frequencies of names 
most_common_genres = X.explode('genres').genres.value_counts() # Create an ordered series of the frequencies of genres 

# The amounts written in the head methods were experimentally determined 
top_companies = [x for x in production_company_freq.head(150).index] # get only the top 150 
top_actors = [x for x in Most_common_actors.head(100).index]
top_important_jobs = [x for x in Most_common_important_jobs.head(200).index] 
top_genres = [x for x in most_common_genres.head(50).index] 

for col, top_x in [('production_companies', top_companies), ('cast', top_actors), ('crew', top_important_jobs), ('genres', top_genres)]:
    for feature in top_x:
        X[feature] = X[col].apply(lambda instance: 1 if feature in instance else 0) # for every name/genre/company, make a new one hot encoded column (feature)
    X = X.drop([col], axis=1) 
X.homepage = X.homepage.apply(has_webpage) # Convert the homepage column into boolean 1 and 0 

rev = X.revenue # All the revenues 
X_training = X.drop(['revenue'], axis=1) # everything but the revenue


# ======== training ========
# Found randomforest reggressor as the most consistent and best
RFReg = RandomForestRegressor(n_estimators=300, max_features='log2') # After using RandomizedSearchCV, these seem to be the best parameters
k_best = SelectKBest(score_func=f_regression, k=370) # number of top features was experimentally determined 
k_best.fit(X_training, rev)
X_train_skb = k_best.transform(X_training)
RFReg.fit(X_train_skb, rev)

# Create a dictionary which keeps track of which features made it to the top 370
k_best_features = {'production_companies':[], 'cast':[], 'crew':[], 'genres':[], 'homepage':False, 'cast_size':False}
for included, feature in zip (k_best.get_support(), X_training.columns):
    if included:
        if feature in top_companies:
            k_best_features["production_companies"].append(feature)
        elif feature in top_actors:
            k_best_features["cast"].append(feature)
        elif feature in top_important_jobs:
            k_best_features["crew"].append(feature)
        elif feature in top_genres:
            k_best_features['genres'].append(feature)
        elif feature == 'homepage':
            k_best_features['homepage'] = True
        elif feature == 'cast_size':
            k_best_features['cast_size'] = True


# ====== Validation stage =======
# preprocessing
X_val = testing_df.loc[:, ['movie_id', 'cast', 'crew', 'genres', 'homepage', 'production_companies', 'revenue']]
X_val.loc[:,['cast', 'crew', 'production_companies', 'genres']] = X_val.loc[:,['cast', 'crew', 'production_companies', 'genres']].applymap(to_json) # convert string to json
X_val.production_companies = X_val.production_companies.apply(production_company_names) # converts messy json list to just list of company names   
X_val.cast = X_val.cast.apply(cast_names) # converts messy json list to just list of cast names
X_val['cast_size'] = X_val.cast.apply(lambda cast: len(cast)) # makes new column with the size of each cast
X_val.crew = X_val.crew.apply(find_important_jobs) # converts messy json list to just list of crew names
X_val.genres = X_val.genres.apply(find_genres) # converts messy json list to just list of grenes

# For every feature in k_best_features, make a new column in testing dataframe - in order to get both training and testing dataframes to have the same dimensions
# e.g. randomforest model gets training on a dataframe of size (2000, 370), the model therefore can only predict models of dimenions (X, 370) 
# so this for loop adds a new 370 columns to testing df, all one hot encoded as to whether that particular name/genre/company from testing df is present in k_best_features
for col in k_best_features:
    if col == 'homepage':
        if k_best_features['homepage']:
            X_val.homepage = X_val.homepage.apply(has_webpage)
        else:
            X_val = X_val.drop(['homepage'], axis=1)
        continue
    elif col == 'cast_size':
        if k_best_features['cast_size']:
            pass
        else:
            X_val = X_val.drop(['cast_size'], axis=1)
        continue
    for feature in k_best_features[col]:
        X_val[feature] = X_val[col].apply(lambda instance: 1 if feature in instance else 0)

# ======== prediction ================
true_rev = X_val.revenue # True revenues
movie_ids = X_val.movie_id 
X_val = X_val.drop(['cast', 'crew', 'production_companies', 'genres', 'movie_id', 'revenue'], axis=1)
y_pred = RFReg.predict(X_val) 
correlation = np.corrcoef(true_rev, y_pred)
mse = mean_squared_error(true_rev, y_pred)

with open('MSE_correlation.csv', 'w') as p1:
    p1_writer = csv.writer(p1)
    p1_writer.writerow(['zid', 'MSE', 'correlation'])
    p1_writer.writerow(['z5017350', f'{mean_squared_error(true_rev, y_pred):.2f}', f'{np.corrcoef(true_rev, y_pred)[0][1]:.2f}'])

with open('revenue_predictions.csv', 'w') as p1:
    p1_writer = csv.writer(p1)
    p1_writer.writerow(['movie_id', 'predicted_revenue'])
    for movieid, predicted_rev in zip(movie_ids, y_pred):
        p1_writer.writerow([movieid, int(predicted_rev)])


# ======== code used for finding some hyper-parameter ========

# param_grid = {"n_estimators":[100, 200, 300] ,"max_features":['sqrt', 'log2'], "min_samples_leaf":[1, 2, 3, 4], "max_depth":[None, 5, 10, 20]}
# grid = RandomizedSearchCV(RFReg, param_grid, cv=10, scoring='neg_root_mean_squared_error')
# k_best = SelectKBest(score_func=f_regression, k=300)
# k_best.fit(X_training, rev)
# X_train_skb = k_best.transform(X_training)
# grid.fit(X_train_skb, rev)
# print(grid.best_score_)
# print(grid.best_params_)
