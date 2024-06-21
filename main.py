#===================Importing dependencies=================================#
import string
import numpy as np
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans, MiniBatchKMeans

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from tqdm.notebook import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


from sklearn.preprocessing import MinMaxScaler

from flask import Flask
pd.options.display.memory_usage = 'deep'



dfmain = pd.read_csv('final_dataset.csv', encoding='latin1') #encoding because of special chars
dfmain['Id'] = range(len(dfmain))
cols = ['Id'] + [col for col in dfmain.columns if col != 'Id']
dfmain = dfmain[cols]

#plt.style.use('ggplot')
#======================Parameter A: Verification Check=====================#
verified = dfmain['verified']

verified_dict = []
for i in range(len(verified)):
    id = i
    verified_dict.append({'Id':id, 'verification_score':verified[i]})
    

verified = pd.DataFrame(verified_dict)
verified['invert_verification_score'] = 1 - verified['verification_score']  # 0 for verified, 1 for not



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

anomaly_score_dict = []
for i in range(len(anomalies)):
    id = i
    score = 1 if anomalies[i] else 0 
    anomaly_score_dict.append({'Id':id, 'anomaly_score':score})
    

anomaly_score = pd.DataFrame(anomaly_score_dict)



#print(fake_reviews)

#======================Parameter C: Classifier=============================#


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
#print(f"Model accuracy: {accuracy_score(y_test, clf.predict(x_test))}")

arr_classifier = []
for i in range(len(dfmain)):
    to_classify = dfmain['review'].iloc[i].lower()
    to_classify = to_classify.translate(str.maketrans('','',string.punctuation)).split()
    to_classify = [stemmer.stem(word) for word in to_classify if word not in stopwords_set]
    to_classify = ' '.join(to_classify)
    review_corpus = [to_classify]
    x_review = vectorizer.transform(review_corpus)

    classifier_score = clf.predict(x_review)
    arr_classifier.append(classifier_score[0])
    
classifier_dict = []
for i in range(len(arr_classifier)):
    id = i
    classifier_dict.append({'Id':id, 'classifier_score':arr_classifier[i]})
    

classifier = pd.DataFrame(classifier_dict)

#======================Parameter D: Sentiment Analysis=============================#
#ax = dfmain['ratings'].value_counts().sort_index().plot(kind='bar', title='Review Count by Stars', figsize = (10,5))

#ax.set_xlabel('review star')
#ax.set_ylabel('Number')
#plt.show()
pd.set_option('display.max_columns', None)  # Show all columns 
pd.set_option('display.width', 1000)
sia = SentimentIntensityAnalyzer()
res = {}
Id=[]
for i in range(len(dfmain)):
    text = dfmain['review'].iloc[i]
    id = i
    Id.append(id)
    res[id] = sia.polarity_scores(text)

dfmain['Id'] = range(len(dfmain))
vaders_result = pd.DataFrame(res).T
vaders_result['Id'] = range(len(dfmain))
vaders_result = vaders_result.merge(dfmain, how='left')

vaders_result['normalized_sentiment_score'] =  vaders_result['compound'].apply(lambda x: 1 if abs(x) > 0.85 else 0)

#ax = sns.barplot(data = vaders_result, x = 'ratings', y='compound')
#ax.set_title('vader plot')
#plt.show()
#======================Parameter E: Helpful tag=========================================#
scaler = MinMaxScaler()
dfmain['helpful_score'] = scaler.fit_transform(dfmain[['helpful']])
dfmain['helpful_score'] = 1 - dfmain['helpful_score']


#======================================Main Function===========================================#

weights = {
    'verification_check' : 0.25,   
    'Autoencoder' : 0.2,         
    'Classifier' : 0.25,          
    'Sentiment_Score' : 0.20,
    'Helpful_Score' : 0.10,
}


dfmain['FINAL_SCORE'] = (
    verified['invert_verification_score'] * weights['verification_check'] +
    anomaly_score['anomaly_score'] * weights['Autoencoder'] +
    classifier['classifier_score'] * weights['Classifier'] +
    vaders_result['normalized_sentiment_score'] * weights['Sentiment_Score'] +
    dfmain['helpful_score'] * weights['Helpful_Score'] 
)



dfmain['FINAL_SCORE'] = (dfmain['FINAL_SCORE'])*10000
dfmain['FINAL_SCORE'] = dfmain['FINAL_SCORE']//79.999999999999

#print(dfmain['FINAL_SCORE'].max())

app = Flask(__name__)

@app.route("/")
def index():
    return str(dfmain['FINAL_SCORE'].iloc[-1])

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 8080, debug = True)