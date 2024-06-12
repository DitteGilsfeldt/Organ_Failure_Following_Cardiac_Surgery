#%%
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.preprocessing import LabelEncoder

###################################################################
# Prepare data for chi-squared test
data = # ...

dropped_features = ['Death Within Year'] # Features to drop
target = 'Death Within Year' # Target variable

X = data.drop(dropped_features, axis=1)
y = data[target]

k = 7 # Number of top features to select
###################################################################

# Function definition
def chi_squared_feature_selection(data, target, dropped_features, k=10):
    """
    Perform feature selection using the Chi-Squared test.

    Parameters:
    data (pd.DataFrame): The dataset containing features and the target variable.
    target (str): The name of the target variable column.
    k (int): The number of top features to select.

    Returns:
    pd.DataFrame: A DataFrame containing the top k features and their Chi-Squared scores.
    """
    # Encode categorical features if they exist
    data_encoded = data.copy()
    for column in data_encoded.select_dtypes(include=['object']).columns:
        data_encoded[column] = LabelEncoder().fit_transform(data_encoded[column])
    
    X = data_encoded.drop(dropped_features + [target], axis=1)
    y = data_encoded[target]
    
    # Apply SelectKBest class to extract top k best features
    bestfeatures = SelectKBest(score_func=chi2, k=k)
    fit = bestfeatures.fit(X, y)
    
    # Get the scores for each feature
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    
    # Concatenate the dataframes for better visualization
    featuresScores = pd.concat([dfcolumns, dfscores], axis=1)
    featuresScores.columns = ['Specs', 'Score']  # Naming the dataframe columns
    
    return featuresScores.nlargest(k, 'Score')

top_features = chi_squared_feature_selection(data, target, dropped_features, k)
print(top_features)
