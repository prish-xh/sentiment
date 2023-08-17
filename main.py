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
#data processing 1/2: remove irrelevant information (punctuation), turn everything to lowercase
import string

def removePunctuation(message):
  # for char in string.punctuation:
  message = message.translate(str.maketrans('', '', string.punctuation))
  return message

train_frame['Message'] = train_frame['Message'].str.lower()
train_frame.head()

# data processing 2/2: install nltk, a library that contains stop words- irrelevant words to ML like "and", "or", etc.
!pip install nltk
import nltk.corpus
from nltk.corpus import stopwords

train_frame['Message']=train_frame['Message'].fillna("")
nltk.download('stopwords')

stop_words = stopwords.words('english')
train_frame['title_nostop'] = train_frame['Message'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

#change the dataframe so it only includes relevant information

train_frame.drop(['Message', 'x'], axis = 1, inplace=True)

train_frame.columns =['sentiment', 'message']

train_frame.shape

#split data into 80/20 train/test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_frame.message, train_frame.sentiment, test_size = 0.2, random_state = 1)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
print(X_train.shape)
print(X_test.shape)

#import count vectorizer (similar to bag of words technique) and transforms data into something that the model can understand

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b\w+\b')
vectorizer.fit(list(X_train) + list(X_test))
x_train_vectorized = vectorizer.transform(X_train)
x_test_vectorized = vectorizer.transform(X_test)

#create and train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000)
model.fit(x_train_vectorized, y_train)

#test the model with the testing data and getting the accuracy from it
from sklearn.metrics import accuracy_score
preds = model.predict(x_test_vectorized)
print(accuracy_score(list(y_test), preds))
