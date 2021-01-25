import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# Load the dataset
MUSIC_PATH = "SpotifyDataset/data.csv"

def load_music_data(music_path = MUSIC_PATH):
    csv_path = os.path.join(music_path)
    return pd.read_csv(csv_path)

music = load_music_data()

print("Data has loaded successfully.")


# As the popularity of a given song is influenced by the "artists", we can create
# a new attribute to take this into account.

# Create an attribute representing the mean popularity of an artists songs

class Artist: 
    def __init__(self, name, popularity): 
        self.name = name
        self.popularity = popularity
        
        
class Track: 
    def __init__(self, name, artists, popularity): 
        self.name = name
        self.artists = artists
        self.popularity = popularity   

tracks = []

names = music.name.values
artists_names = music.artists.values
popularity = music.popularity.values

print("Creating new attribute ""artist_popularity""...")

for index in range(len(names)): 
    track = Track(names[index], artists_names[index], popularity[index])
    tracks.append(track)
    
    
artists = []
artists_names_done = []
artists_popularities = []

for artists_str in artists_names: 
    artists_sub_list = artists_str[1:-1].split(', ')
    
    track_pop = 0
    for artist in artists_sub_list: 
        
        if artist in artists_names_done: 
            a = [x for x in artists if x.name == artist][0]
            artist_pop = a.popularity
            
        else: 
            songs_pop = [x.popularity for x in tracks if artist in x.artists]
            artist_pop = sum(songs_pop) / len(songs_pop)
            artists_names_done.append(artist)
            a = Artist(artist, artist_pop)
            artists.append(a)
        
        track_pop += artist_pop
        
    track_pop /= len(artists_sub_list)
    artists_popularities.append(track_pop)
    
artists_popularities = np.asarray(artists_popularities)

music["artist_popularity"] = artists_popularities

print("New attrivute ""artist_popularity"" added successfully.")

# Creating the test set and train set
train_set, test_set = train_test_split(music, test_size=0.2, random_state=42)


# Revert to a clean training set and separate predictors and labels
music_labels = train_set["popularity"].copy()
music_set = train_set.drop("popularity", axis=1)

# Also need to drop "release_date", "id", "name" and "artists"
music_set = music_set.drop("release_date", axis=1)
music_set = music_set.drop("id", axis=1)
music_set = music_set.drop("name", axis=1)
music_set = music_set.drop("artists", axis=1)

# Transformation Pipelines and Feature Scaling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

music_prepared = num_pipeline.fit_transform(music_set)

# Building a Random Forest Regressor Model
print("Training a Random Forest Regressor Model...")

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(music_prepared, music_labels)


# Preparing the test set so we can check the model against it.
test_set = test_set.drop("release_date", axis=1)
test_set = test_set.drop("id", axis=1)
test_set = test_set.drop("name", axis=1)
test_set = test_set.drop("artists", axis=1)

y_test = test_set["popularity"].copy()
x_test = test_set.drop("popularity", axis=1)

# Evaluate the model on the test set
final_model = forest_reg

x_test_prepared = num_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

# Can print out the first 100 entries to check the performance of the model 
for index in range(len(final_predictions[:100])): 

    pred = final_predictions[index]
    actual = y_test.iloc[index]
    
    print("Actual / Predicted: {:.4f} / {:.4f}".format(actual, pred))

# Save the model
filename = "forest_reg_popularity_model.pkl"
joblib.dump(final_model, filename)


