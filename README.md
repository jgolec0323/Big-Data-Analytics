# Big-Data-Analytics-Semester-Project
Semester Project for  Big Data Analytics at the University of Iowa

# Use Case 

## Data Understanding
Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Header: "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"

id: Column to represent the unique identifier for the user

comment_text: string of data that represents the comment

toxic,severe_toxic,obscene,threat,insult,identity_hate: Binary labels for the comment. Comment can have multiple labels.

## Data Preprocessing

### Data Cleansing 

The first step, preprocessing, is mainly cleaning the data by removing any unnecessary and irrelevant words. To clean the data,
- Removed contractions (e.g. can’t to can not).
- Converted all words to lowercase
- Removed numbers, usernames, and IP addresses.
- Removed stop words (e.g. “and”, “or”, “the”, “that”, etc.)

## Exploratory Analysis


### Feature Generation

TF-IDF (Term Frequency - Inverse Document Frequency)

The term frequency is the frequency of a word in each comment string. The Inverse data frequency is the weight of all words across all comments in the data (i.e. rare words have a high IDF score). Simply, TF-IDF values are associated with words with significance. This function allows the collection of comment strings to be transformed into a numeric vector.

## Modeling 

Multi-label classificiation

Models: 

