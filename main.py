#===================Importing dependencies=================================#
import string
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
pd.options.display.memory_usage = 'deep'

final_score = 0

dfmain = pd.read_csv('final_dataset.csv', encoding='latin1') #encoding because of special chars

intinput = int(input("Enter Input Entry field (0-7651)")) #For now using this, ideally our interface should have something jisme cust inp daalsake#


#======================Parameter A: Verification Check=====================#
if (dfmain['verified'].iloc[intinput])==0:
    final_score+=25

#======================Parameter B: Autoencoder for anomaly detection============================#
texts = dfmain['review'] 
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

max_len = 100  # Define a max length for padding
data_padded = pad_sequences(sequences, maxlen=max_len)

input_dim = data_padded.shape[1]  # Number of features in input

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(data_padded, data_padded, epochs=50, batch_size=256, shuffle=True, validation_split=0.2)


reconstructions = autoencoder.predict(data_padded)
mse = np.mean(np.power(data_padded - reconstructions, 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # You can adjust the percentile based on your needs
anomalies = mse > threshold

# Flag potential fake reviews
fake_reviews = dfmain[anomalies]

print(fake_reviews)



#from k calculation.py, optimal k=6

#======================Parameter C: Classifier=============================#
nltk.download('stopwords')

df = pd.read_csv('Classifier_dataset.csv', usecols = ["text_", "label_num"], dtype={"label_num" : "int8"})


df['text_'] = df['text_'].apply(lambda x: x.replace('\r\n', ' ')) 


#stemming 
stemmer = PorterStemmer()
corpus = []

stopwords_set = set(stopwords.words('english'))

for i in range(len(df)):
    text = df['text_'].iloc[i].lower()
    text = text.translate(str.maketrans('','',string.punctuation)).split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    text = ' '.join(text)
    corpus.append(text)


#vectorize 
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus).toarray()
y = df['label_num']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

clf = MultinomialNB()
clf.fit(x_train, y_train)
print(f"Model accuracy: {accuracy_score(y_test, clf.predict(x_test))}")


#======================Parameter D: Sentiment Analysis=============================#

#======================Parameter E: Helpful tag, Reviews/acc=============================#
#email_to_classify = df.text.values[10]
def finrev():
    print(final_score)
    





