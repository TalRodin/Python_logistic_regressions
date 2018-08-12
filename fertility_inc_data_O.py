from plotly.offline import  plot
import plotly.graph_objs as go
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import plotly.plotly as py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from plotly.figure_factory import create_table
from plotly.graph_objs import Scatter, Figure, Layout
import scipy
import time
from plotly.grid_objs import Grid, Column
import plotly
import json
import csv
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas_datareader.data as web
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#Input variables
#Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1) nominal categorical data
#Age at the time of analysis. 18-36 (0, 1) 
#Childish diseases (ie , chicken pox, measles, mumps, polio)	1) yes, 2) no. (0, 1)  nominal categorical data
#Accident or serious trauma 1) yes, 2) no. (0, 1) nominal categorical data
#Surgical intervention 1) yes, 2) no. (0, 1) nominal categorical data
#High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1) ordinal categorical data
#Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1), ordinal categorical data 
#Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1) ordinal categorical data
#A Number of hours spent sitting per day ene-16	(0, 1) 
#Output: Diagnosis	normal (N), altered (O)	

#Predict y - will the diagnosis be normal or altered (binary)

def main():
    #downloaded the data from .csv file
    data_F=pd.read_csv('/users/alyonarodin/Desktop/fertility.csv')
    print(data_F)
    #deleted NA values
    #print(data_F.isnull().sum())
    #data_F=data_F.dropna()
    
     #printed the the first few rows of the table to see
    print(data_F.head())
    
    #plotting amount N vs O
    count_N,count_O=plot_N_O(data_F)
    
    Season=add_data(data_F['Season'],count_N,count_O,data_F)
    Age=add_data(data_F['Age'],count_N,count_O,data_F)
    Childish_diseases=add_data(data_F['Childish diseases'],count_N,count_O,data_F)
    Accident=add_data(data_F['Accident'],count_N,count_O,data_F)
    Surgical_intervention=add_data(data_F['Surgical intervention'],count_N,count_O,data_F)
    High_fevers=add_data(data_F['High fevers'],count_N,count_O,data_F)
    Frequency=add_data(data_F['Frequency'],count_N,count_O,data_F)
    Smoking=add_data(data_F['Smoking'],count_N,count_O,data_F)
    sitting=add_data(data_F['sitting'],count_N,count_O,data_F)
    Diagnosis=add_data(data_F['Diagnosis'],count_N,count_O,data_F) 
    
    df_O_data=pd.DataFrame(list(zip(Season,Age,Childish_diseases,Accident,Surgical_intervention,High_fevers,Frequency,Smoking,sitting,Diagnosis)),columns=['Season','Age','Childish diseases','Accident','Surgical intervention','High fevers','Frequency','Smoking','sitting','Diagnosis'])
   
 
    n_data=pd.concat([data_F, df_O_data,df_O_data,df_O_data,df_O_data,df_O_data,df_O_data])
    
    new_data=shuffle_data(n_data)
    print(new_data.head())

    count_N,count_O=plot_N_O(new_data)
def plot_N_O(data_F):
    
    count_N=0
    count_O=0
    for i in data_F['Diagnosis']:
        if i=='N':
            count_N+=1
        
        else:
            count_O+=1
    print('N: ',count_N)
    print('O: ',count_O)
    return(count_N,count_O)
    
    #or

    data_F['Diagnosis'].value_counts()
    sns.countplot(x='Diagnosis', data=data_F, palette="coolwarm")
    plt.show()

#since we have unequival data - outcome - solution to increse the data of N
#we increase by 7 times
def add_data(x,count_N,count_O,data_F):
 list_=[]
 if count_N > count_O:
    for i in range(len(data_F['Diagnosis'])):
        if data_F['Diagnosis'][i]=='O':

            list_.append(x[i])
        
 return(list_)

def shuffle_data(new_data):
    new_data = shuffle(new_data, random_state=0)
    return(new_data)

main()    
    
    
    
    
    
    
    
    
    
