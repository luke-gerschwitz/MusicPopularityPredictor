# MusicPopularityPredictor

This project aims to develop a machine learning model to accurately predict the "popularity" of a song given all the other song metrics. It will be using the "Spotify Dataset 1921-2020, 160k+ Tracks" dataset found on Kaggle (https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks). This is a dataset containing over 160,000 songs collected from the Spotify Web API.

The Spotify Web API provides various metrics for the songs on its service (i.e. "danceability", "tempo", etc). The "popularity" metric is a value between 0-100 which measures the popularity of a given song. 

  # Visualising the Data

The following paragraphs will be a brief overview of the process taken to visualise the dataset and prepare it for the machine learning algorithms. Jupyter Notebook is used for this process. The code can be found in the JupyterNotebook folder. 

Firstly the data has to be loaded:

```
import os
import numpy as np
import pandas as pd

MUSIC_PATH = "../SpotifyDataset/data.csv"

def load_music_data(music_path = MUSIC_PATH):
    csv_path = os.path.join(music_path)
    return pd.read_csv(csv_path)
    
music = load_music_data()    
```

We can take an initial look at the dataset's attributes:

```
>>> music.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 170653 entries, 0 to 170652
Data columns (total 19 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   valence           170653 non-null  float64
 1   year              170653 non-null  int64  
 2   acousticness      170653 non-null  float64
 3   artists           170653 non-null  object 
 4   danceability      170653 non-null  float64
 5   duration_ms       170653 non-null  int64  
 6   energy            170653 non-null  float64
 7   explicit          170653 non-null  int64  
 8   id                170653 non-null  object 
 9   instrumentalness  170653 non-null  float64
 10  key               170653 non-null  int64  
 11  liveness          170653 non-null  float64
 12  loudness          170653 non-null  float64
 13  mode              170653 non-null  int64  
 14  name              170653 non-null  object 
 15  popularity        170653 non-null  int64  
 16  release_date      170653 non-null  object 
 17  speechiness       170653 non-null  float64
 18  tempo             170653 non-null  float64
dtypes: float64(9), int64(6), object(4)
memory usage: 24.7+ MB
```

Fortunately there are no null entries in the dataset. However, will need to make adjustments to the text attributes "artists", "id", "name" and "release_date".

We can plot the data for all the numerical attributes to get a visualisation of the range of values:

```
%matplotlib inline
import matplotlib.pyplot as plt
music.hist(bins=50, figsize=(20,15))
plt.show()
```

![Histograms](Images/AttributeHistograms.png?raw=true "Histograms for Numerical Attributes")


We can see that the majority of songs have a popularity score of 0 (Over 30,000 songs), with the remainder of the songs having an average popularity score of around 35-45.

As seen below, there is a clear correlation between the year a song releases and the popularity, with songs in more recent years generally being more popular.

```
music.plot(kind="scatter", x="popularity", y="year", alpha=0.1)
```

![Year-Popularity Correlation](Images/Popularity-YearVisualisation.png?raw=true "Correlation between Year and Popularity")


We can gain slighty more insight into the correlation of the attributes as seen below. We can see that generally the higher energy and louder a song is, the more popular it will be.

```
# x-axis is energy, y-axis is loudness.
# The radius of each circle represents the danceability of a song, with the colormapping for a songs popularity

music_set.plot(kind="scatter", x="energy", y="loudness", alpha=0.4,
              s=music_set["danceability"], label="danceability", figsize=(13,10),
              c="popularity", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
```

![Energy-Loudness-Popularity Visualisation](Images/Energy-Loudness-PopularityVisualisation.png?raw=true)


  # Data Cleanup and Preparation

We can also look at how each attribute correlates to the "popularity" attribute:

```
>>>corr_matrix = music_set.corr()
>>>corr_matrix["popularity"].sort_values(ascending=False)  

popularity          1.000000
year                0.861550
energy              0.484203
loudness            0.456886
danceability        0.200023
explicit            0.193704
tempo               0.131884
duration_ms         0.059956
valence             0.014147
key                 0.005593
mode               -0.028449
liveness           -0.075825
speechiness        -0.169889
instrumentalness   -0.297421
acousticness       -0.572544
Name: popularity, dtype: float64
```

As previously discussed, attributes "year", "energy" and "loudness" do have a strong positive correlation to the popularity (In particular the "year" attribute). It would be benefitial to make use of the "artists" attribute, as a song from a well-known artist is always going to be more popular than a song from an unkown artist. Therefore, we can create a new attribute to represent the mean popularity of an artists songs. Refer to file "music_popularity_model.py" to see how this was created.


Now that we have all the attributes we need to train the model, we can create a training set and test set:


```
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(music, test_size=0.2, random_state=42)
music_set = train_set.copy()
music_labels = train_set["popularity"].copy()
music_set = train_set.drop("popularity", axis=1)
```


We also need to remove the non-numerical attributes from the dataset in preparation for training the model.

```
music_set = music_set.drop("release_date", axis=1)
music_set = music_set.drop("id", axis=1)
music_set = music_set.drop("name", axis=1)
music_set = music_set.drop("artists", axis=1)
```

Now that the dataset is ready, we can create the transformation pipeline:

```
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

music_prepared = num_pipeline.fit_transform(music_set)
```

  # Selecting and Training a Machine Learning Model

This section will purely focus on training the Random Forest Regressor. However I did test using a Linear Regression model and Decision Tree Regressor model, however they both had inferior performance results. Please refer to the models folder and Jupyter Notebook file for the training code and evaluation.

Haing completed the data preparation, cleanup and transformation pipeline, training the Random Forest Regressor is relatively simple.

```
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(music_prepared, music_labels)
```


Having trained the model, we can look at some of the results. 

```
from sklearn.model_selection import cross_val_score

music_predictions = forest_reg.predict(music_prepared)
forest_mse = mean_squared_error(music_labels, music_predictions)
forest_rmse = np.sqrt(forest_mse)

forest_scores = cross_val_score(forest_reg, music_prepared, music_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```
```
Scores: [8.1608569  8.09987402 8.02784611 8.04598308 8.22075479 8.10091806
 8.1587252  8.15181274 8.22549239 8.11276108]
Mean: 8.130502436862113
Standard deviation: 0.06257040900770887
```

  # Evaluating Model on Test Set
  
First prepare the test set:

```
test_set = test_set.drop("release_date", axis=1)
test_set = test_set.drop("id", axis=1)
test_set = test_set.drop("name", axis=1)
test_set = test_set.drop("artists", axis=1)

y_test = test_set["popularity"].copy()
x_test = test_set.drop("popularity", axis=1)
```

Can now evaluate the model on the test:

```
final_model = forest_reg
x_test_prepared = num_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
```

The model has now made the predictions on the test set, lets look at the results:

```
>>>average_error = (abs(y_test - final_predictions)).mean()
>>>print("{:.4f} average error".format(average_error))
5.4850 average error
```

Finally, to get a sense of how well the model is actually performing, we can print out the predictions and results of the first 100 songs in the test set:

```
for index in range(len(final_predictions[:100])): 

    pred = final_predictions[index]
    actual = y_test.iloc[index]
    
    print("Actual / Predicted: {:.4f} / {:.4f}".format(actual, pred))
```

```
Actual / Predicted: 34.0000 / 34.6575
Actual / Predicted: 26.0000 / 32.4025
Actual / Predicted: 38.0000 / 41.7600
Actual / Predicted: 13.0000 / 18.2300
Actual / Predicted: 0.0000 / 0.0300
Actual / Predicted: 62.0000 / 64.5800
Actual / Predicted: 23.0000 / 24.4120
Actual / Predicted: 0.0000 / 0.6600
Actual / Predicted: 25.0000 / 25.8400
Actual / Predicted: 25.0000 / 33.0800
Actual / Predicted: 48.0000 / 52.5150
Actual / Predicted: 42.0000 / 45.7000
Actual / Predicted: 35.0000 / 38.0200
Actual / Predicted: 22.0000 / 28.2800
Actual / Predicted: 34.0000 / 36.4200
Actual / Predicted: 49.0000 / 38.2500
Actual / Predicted: 0.0000 / 0.0700
Actual / Predicted: 46.0000 / 45.4800
Actual / Predicted: 50.0000 / 50.8100
Actual / Predicted: 33.0000 / 22.3333
Actual / Predicted: 45.0000 / 43.1600
Actual / Predicted: 8.0000 / 16.3200
Actual / Predicted: 22.0000 / 18.6600
Actual / Predicted: 62.0000 / 66.7300
Actual / Predicted: 20.0000 / 29.9500
Actual / Predicted: 29.0000 / 32.9300
Actual / Predicted: 51.0000 / 40.3800
Actual / Predicted: 35.0000 / 26.5600
Actual / Predicted: 56.0000 / 58.4850
Actual / Predicted: 34.0000 / 45.5400
Actual / Predicted: 37.0000 / 39.7200
Actual / Predicted: 24.0000 / 22.4600
Actual / Predicted: 38.0000 / 26.4800
Actual / Predicted: 31.0000 / 40.7667
Actual / Predicted: 41.0000 / 45.6900
Actual / Predicted: 15.0000 / 23.4720
Actual / Predicted: 27.0000 / 35.9500
Actual / Predicted: 0.0000 / 0.0000
Actual / Predicted: 26.0000 / 39.4213
Actual / Predicted: 0.0000 / 2.4350
Actual / Predicted: 46.0000 / 22.2900
Actual / Predicted: 5.0000 / 15.5800
Actual / Predicted: 0.0000 / 1.8100
Actual / Predicted: 55.0000 / 58.9833
Actual / Predicted: 0.0000 / 3.2150
Actual / Predicted: 0.0000 / 0.1000
Actual / Predicted: 23.0000 / 30.0900
Actual / Predicted: 54.0000 / 38.2600
Actual / Predicted: 37.0000 / 48.4367
Actual / Predicted: 26.0000 / 30.6600
Actual / Predicted: 40.0000 / 36.3900
Actual / Predicted: 11.0000 / 21.5200
Actual / Predicted: 30.0000 / 43.8800
Actual / Predicted: 33.0000 / 37.2650
Actual / Predicted: 14.0000 / 20.7867
Actual / Predicted: 58.0000 / 53.4100
Actual / Predicted: 0.0000 / 0.1200
Actual / Predicted: 0.0000 / 0.0000
Actual / Predicted: 0.0000 / 2.1683
Actual / Predicted: 56.0000 / 37.4100
Actual / Predicted: 46.0000 / 55.1500
Actual / Predicted: 63.0000 / 67.4033
Actual / Predicted: 12.0000 / 13.3820
Actual / Predicted: 9.0000 / 7.7000
Actual / Predicted: 37.0000 / 44.6150
Actual / Predicted: 41.0000 / 51.5200
Actual / Predicted: 40.0000 / 40.7160
Actual / Predicted: 50.0000 / 55.6200
Actual / Predicted: 67.0000 / 52.9400
Actual / Predicted: 19.0000 / 30.5800
Actual / Predicted: 52.0000 / 60.2114
Actual / Predicted: 56.0000 / 55.0167
Actual / Predicted: 2.0000 / 1.5300
Actual / Predicted: 5.0000 / 14.8100
Actual / Predicted: 0.0000 / 0.6933
Actual / Predicted: 0.0000 / 0.0000
Actual / Predicted: 39.0000 / 50.3300
Actual / Predicted: 45.0000 / 44.5300
Actual / Predicted: 0.0000 / 0.0000
Actual / Predicted: 50.0000 / 54.0000
Actual / Predicted: 0.0000 / 0.0000
Actual / Predicted: 64.0000 / 56.3450
Actual / Predicted: 53.0000 / 35.8150
Actual / Predicted: 44.0000 / 37.6300
Actual / Predicted: 0.0000 / 0.0000
Actual / Predicted: 0.0000 / 0.0000
Actual / Predicted: 35.0000 / 39.0400
Actual / Predicted: 34.0000 / 36.0000
Actual / Predicted: 22.0000 / 31.0300
Actual / Predicted: 51.0000 / 49.4100
Actual / Predicted: 39.0000 / 44.4300
Actual / Predicted: 41.0000 / 51.4900
Actual / Predicted: 52.0000 / 50.9200
Actual / Predicted: 35.0000 / 38.4500
Actual / Predicted: 29.0000 / 31.5467
Actual / Predicted: 0.0000 / 0.2000
Actual / Predicted: 2.0000 / 1.4300
Actual / Predicted: 48.0000 / 43.2200
Actual / Predicted: 29.0000 / 26.8200
Actual / Predicted: 48.0000 / 56.7033
```
