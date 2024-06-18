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

from sklearn.cluster import KMeans

pd.options.display.memory_usage = 'deep'

final_score = 0

dfmain = pd.read_csv('final_dataset.csv', encoding='latin1') #encoding because of special chars

intinput = int(input("Enter Input Entry field (0-7651)")) #For now using this, ideally our interface should have something jisme cust inp daalsake#


#======================Parameter A: Verification Check=====================#
if (dfmain['verified'].iloc[intinput])==0:
    final_score+=25

#======================Parameter B: NLP Analysis============================#

                        #Part B1 -> K-mean clustering 
documents = dfmain['review'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)
k = 20
model = KMeans(n_clusters=k, init = 'k-means++', max_iter= 100, n_init=1)
model.fit(features)


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




#email_to_classify = df.text.values[10]
def finrev():
    print(final_score)






