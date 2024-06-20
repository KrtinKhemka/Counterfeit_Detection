import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

dfmain = pd.read_csv('final_dataset.csv', encoding='latin1')

#ax = dfmain['ratings'].value_counts().sort_index().plot(kind='bar', title='Review Count by Stars', figsize = (10,5))

#ax.set_xlabel('review star')
#ax.set_ylabel('Number')
#plt.show()

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
#print(vaders_result['compound'])
ax = sns.barplot(data = vaders_result, x = 'ratings', y='compound')
ax.set_title('vader plot')
plt.show()
