# HackerNewsClassification
Comp 472 Assignment 2 - Hacker News Naïve Bays Classifier

# Python version 
Only tested with python version Python 3.6.6
I am assuming this program will run on version higher than 3.6.6

# Dependencies
Please view the requirements.text to view the dependencies need to run.
Note the minimum dependencies used are the following

- pandas~=1.0.4
- matplotlib~=3.1.1
- sklearn~=0.0
- scikit-learn~=0.23.1
- nltk~=3.5

# Running the application

I only ran the application through pycharm. In pycharm you must create a virtual environment
to do this please follow the following instructions

`https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html`

To use the requirements.txt in the environment please follow the instructions below
`https://www.jetbrains.com/help/idea/managing-dependencies.html` 


# Using the application

only run main.py

depending on computer speeds it should take around 1-5 minutes to run all the classifications and show the performance
graphs

the results folder will contain many .txt files once the classifications run

these files contain information about the classification experiments.

the files with a '-model' in the file name are the actual trained models for the 2018 training data
the 2019 data is tested against the model.

the files with '-results' in the filename contain the results of the classifications for a given 
experiment.

## Test files
test.csv is a subset of hns_2018-2019.csv it contains 200 rows of even data. This was use during development because
of iteration speeds.

hns_2018-2019.csv is the full data set comprising of 10 000 rows


# Sample output

![ScreenShot 1](images/screen_shot_1.png?raw=true)






