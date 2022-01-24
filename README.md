# Big-Data-Analytics-Semester-Project
Semester Project for  Big Data Analytics at the University of Iowa

# Use Case 

## Data Understanding
Source: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Header: "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"

id: Column to represent the unique identifier for the user

comment_text: string of data that represents the comment

toxic,severe_toxic,obscene,threat,insult,identity_hate: Binary labels for the comment. Comment can have multiple labels.

## Data Prep

### Data Cleansing 

The first step, preprocessing, is mainly cleaning the data by removing any unnecessary and irrelevant words. To clean the data,
- Removed contractions (e.g. can’t to can not).
- Converted all words to lowercase
- Removed numbers, usernames, and IP addresses.
- Removed stop words (e.g. “and”, “or”, “the”, “that”, etc.)

### Feature Generation

Term Frequency - Inverse Document 

## Exploratory Analysis

## Modeling 

