%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

# original models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# new models
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

# ensemble methods 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

# import training file 
train = '/Users/johngolec/Documents/Fall 2018/Big Data Analytics/Semester Project/all/train.csv'
df = pd.read_csv(train, encoding = "ISO-8859-1")

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# clean the comment strings
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

# clean comment_text column
df['comment_text'] = df['comment_text'].map(lambda com : clean_text(com))

#split into train and test data and create dataframes for train and test only including the comment_text column
train, test = train_test_split(df, random_state=42, test_size=0.33)
X_train = train.comment_text
X_test = test.comment_text

# Function to train a pipeline by creating a classifier for each toxic category and printing the roc-auc score and f1 score metric, as well as the testing accuracy, confusion matrix, and classification report. At the end, it prints a table of the roc-auc score and f1 score for each category.
def MultiLabelTrain(pipeline):
    accuracies = []
    roc_auc_scores = []
    f1_scores = []
    metrics = ['roc-auc score','f1 scores']
    for category in categories:
        print('... Processing {}'.format(category))
        pipeline.fit(X_train, train[category])
        prediction = pipeline.predict(X_test)
        score = accuracy_score(test[category], prediction)
        
        roc_auc = roc_auc_score(test[category], prediction)
        roc_auc_scores.append(roc_auc)
        print('ROC-AUC Score is {}'.format(roc_auc))
        
        f1 = f1_score(test[category], prediction)
        f1_scores.append(f1)
        print('F1 score is {}'.format(f1))
        
        print('Test accuracy is {}'.format(score))
        print(confusion_matrix(test[category],prediction))
        print(classification_report(test[category],prediction))
        accuracies.append(score)
    print('Average testing accuracy is {}'.format(np.mean(accuracies)))  
    lis = [roc_auc_scores,f1_scores]
    score_df = pd.DataFrame(lis, columns = categories)
    score_df.insert(loc=0, column = 'Metric', value = metrics)
    #averages = [np.mean(roc_auc_scores), np.mean(f1_scores)]
    #score_df.insert(loc=7, column = 'Averages', value = averages]
    print(score_df)

# Naive Bayes
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
            ])
MultiLabelTrain(NB_pipeline)

# Logistic Regression 
LR_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])
MultiLabelTrain(LR_pipeline)

# Support Vector Machines
SVM_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
MultiLabelTrain(SVM_pipeline)

# Ridge Classifier
Ridge_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(RidgeClassifier(), n_jobs=1)),
            ])
MultiLabelTrain(Ridge_pipeline)

# Perceptron Classifier (equivalent to SGD with loss='perceptron', eta0=1, learning_rate=”constant”, penalty=None)
Perceptron_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(Perceptron(), n_jobs=1)),
            ])
MultiLabelTrain(Perceptron_pipeline)

# Stochastic Gradient Descent Classifier (similar to perceptron, except the learning rate is optimal)
SGD_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(SGDClassifier(loss='perceptron', eta0=1, penalty=None), n_jobs=1)),
            ])
MultiLabelTrain(SGD_pipeline)

# Passive Aggressive Classifier 
PAC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(PassiveAggressiveClassifier(), n_jobs=1)),
            ])
MultiLabelTrain(PAC_pipeline)



# Bagging ensemble methods for generalization of the best performing classifiers
# SVM bagging
SVM_bagging = SVM_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(BaggingClassifier(LinearSVC()), n_jobs=1)),
            ])
MultiLabelTrain(SVM_bagging)

# Perceptron bagging
Perceptron_bagging = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(BaggingClassifier(Perceptron()), n_jobs=1)),
            ])
MultiLabelTrain(Perceptron_bagging)

# Stochastic Gradient Descent bagging
SGD_bagging = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(BaggingClassifier(SGDClassifier(loss='perceptron', eta0=1, penalty=None)),n_jobs=1)),
            ])
MultiLabelTrain(SGD_bagging)

# Passive Agressive bagging
PAC_bagging = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(BaggingClassifier(PassiveAggressiveClassifier()), n_jobs=1)),
            ])
MultiLabelTrain(PAC_bagging)
    
#def testData(pipeline)    
#    for category in categories:
#        print('... Processing {}'.format(category))
#        prediction = pipeline.predict(X_test)
#        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))    
    
    

        