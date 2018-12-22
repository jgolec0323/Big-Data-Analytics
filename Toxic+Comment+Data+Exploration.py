import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
from nltk.tokenize import word_tokenize

# download file
train_file = '/Users/johngolec/Documents/Fall 2018/Big Data Analytics/Semester Project/all/train.csv'

train_df = pd.read_csv(train_file, sep = ',')
train_df.head()

#split data into comments and toxic classifications
comments_df = train_df['comment_text']
toxic_df = train_df.drop(['id','comment_text'], axis=1)
comments_df.head()
toxic_df.head()

# PLOT NUMBER 1:
# count number of comments classified in each category and visualize the number of comments classified in each category with a bar chart
counts = []
categories = list(toxic_df.columns.values)
for i in categories:
    counts.append((i, toxic_df[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])

plt.figure(figsize=(8,5))
multilabel_plot = sns.barplot(df_stats['category'],df_stats['number_of_comments'])
multilabel_plot.grid(False)
plt.title("Number of Comments per Category")
plt.ylabel('Number of occurrences', fontsize=12)
plt.xlabel('Toxic category', fontsize=12)


# PLOT NUMBER 2: 
# find how many comments are labeled more than once and plot 
rowsums = toxic_df.sum(axis=1)
x=rowsums.value_counts()

plt.figure(figsize=(8,5))
multilabel_plot = sns.barplot(x.index, x.values)
multilabel_plot.grid(False)
plt.title("Comments with Muptiple Categories")
plt.ylabel('Number of occurrences', fontsize=12)
plt.xlabel('Number of categories', fontsize=12)

# PLOT NUMBER 3:
# find lengths of each comment string and plot histogram of comment lengths
lens = comments_df.str.len()
max_len = np.amax(lens)
min_len = np.amin(lens)
average_len = np.mean(lens)
hist = lens.hist(bins = np.arange(min_len,max_len,50))
hist.set_title('Histogram of comment length')
hist.set_ylabel('Number of occurences')
hist.set_xlabel('Length of comment')
hist.grid(False)

# function to clean comment text
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
    
    text = re.sub('\\n',' ',text)
    text = re.sub("\[\[User.*",'',text)
    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',text)
    text = re.sub("(http://.*?\s)|(http://.*)",'',text)
    
    text = re.sub("(1)|(2)|(3)|(4)|(5)|(6)|(7)|(8)|(9)|(0)|",'',text)
    
    text = text.strip(' ')
    
    # tokenize a sentence and remove stopwords from word_tokens
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    #return clean text
    return filtered_sentence

# map through comment_text column of the training data and clean every comment
train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))

# PLOT NUMBER 4:
# find lengths of each comment string and plot histogram of comment lengths AFTER data cleaning
lens2 = train_df.comment_text.str.len()
max_len2 = np.amax(lens)
min_len2 = np.amin(lens)
average_len2 = np.mean(lens)
hist2 = lens.hist(bins = np.arange(min_len,max_len,50))
hist2.set_title('Histogram of comment length')
hist2.set_ylabel('Number of occurences')
hist2.set_xlabel('Length of comment')
hist2.grid(False)

