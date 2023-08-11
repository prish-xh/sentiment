#import kaggle, add kaggle credentials
! pip install -q kaggle
from google.colab import drive
drive.mount('/content/gdrive')
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

#download dataset from kaggle
!kaggle datasets download -d jp797498e/twitter-entity-sentiment-analysis
!unzip twitter-entity-sentiment-analysis.zip

#import pandas - a library that can read and store the data - and store data as a dataframe
import pandas as pd
train_frame = pd.read_csv('twitter_training.csv')
train_frame.columns =['x', 'Location', 'Sentiment', 'Message']
train_frame.drop(['Location'], axis = 1, inplace=True)

#turn sentiments into numbers, to allow the model to predict it
train_frame.Sentiment = train_frame.Sentiment.map({'Positive':1, 'Negative':2, 'Neutral':0, 'Irrelevant':0})
train_frame.head()
