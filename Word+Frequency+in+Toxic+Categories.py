%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import seaborn as sns
import ntlk
from nltk.tokenize import word_tokenize 

filename = '/Users/johngolec/Documents/Fall 2018/Big Data Analytics/Semester Project/all/train.csv'
df = pd.read_csv(filename, encoding = "ISO-8859-1")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text
    
df['comment_text'] = df['comment_text'].map(lambda com : clean_text(com))

def token(text):    
    word_tokens = word_tokenize(text) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        filtered_sentence = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w)
    return text

df['comment_text'] = df['comment_text'].map(lambda com : token(com))
df['comment_text'] = df['comment_text'].map(lambda com : word_tokenize(com))

toxic = df.loc[df['toxic'] == 1].comment_text
severe_toxic = df.loc[df['severe_toxic'] == 1].comment_text
obscene = df.loc[df['obscene'] == 1].comment_text
threat = df.loc[df['threat'] == 1].comment_text
insult = df.loc[df['insult'] == 1].comment_text
identity_hate = df.loc[df['identity_hate'] == 1].comment_text

# find word frequency and plot bar chart of 30 most common words 
def findWordFreq(data, category):
    flat_list = [item for sublist in data for item in sublist]
    fdist = nltk.FreqDist(flat_list)
    common = fdist.most_common(30)
    plt.bar(range(len(common)), [val[1] for val in common], align='center')
    plt.xticks(range(len(common)), [val[0] for val in common])
    plt.xticks(rotation=70)
    plt.ylabel('Frequency')
    plt.title(category)
    plt.show()

    
# find most frequent words for each toxic category
findWordFreq(toxic, 'Toxic')  
findWordFreq(severe_toxic, 'Severe Toxic')  
findWordFreq(obscene, 'Obscene')  
findWordFreq(threat, 'Threat')          
findWordFreq(insult, 'Insult') 
findWordFreq(identity_hate, 'Identity Hate')  
        
    
    
            