from plotly.offline import  plot
import plotly.graph_objs as go
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
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

    #deleted NA values
    print(data_F.isnull().sum())
    data_F=data_F.dropna()

    #printed the the first few rows of the table to see
    print(data_F.head())
    
    #plotting amount N vs O
    plot_N_O(data_F)
    
    #example how logistic regrestic looks like, plotting sigmoid function
    #x values range from -10 to 10
    z = np.arange(-8, 8, 0.1)
    
    # plug into sigmoid function values x
    phi_z = sigmoid(z)
   
    #plot sigmoid function 
    sigm_plot(z,phi_z)
    
    #to check if any column names has empty spaces that can give an error
    print ("<{}>".format(data_F.columns[9]))
    
    #little prepare data 
    data_F=prep_data(data_F)
    print(data_F.head())
    
    #assigned to X values which are not dummy 
    X = data_F[['Diagnosis_code','Age','High_fevers','Alcohol_consumption','Smoking','Sitting' ]]
    
    #prepared dummy and new dataset
    new_data=dummy_new_table(X,data_F)
    print(new_data.head())
    
    #shuffled data
    new_data=shuffle_data(new_data)
    print(new_data.head())
    
    #checked for correlation
    corr=corr_fun(new_data)
    print(corr)
    
    #droped the base columns for each dummy value
    new_data2=new_data.drop(['Accident_trauma_0'],axis=1)
    new_data2=new_data2.drop(['Childish_diseases_0'], axis=1)
    new_data2=new_data2.drop(['Surgical_intervention_0'], axis=1)
    new_data2=new_data2.drop(['Season_-1.0'], axis=1)
    print(new_data2.head())
    
    new_corr=new_corr_fun(new_data2)
    print(new_corr)
    
    #plot new correlation without multicollinearity
    plotly_new_corr(new_corr)
    
    
    #converted to matrix and returned two matrices A and b (A*x=b)
    data,target= matrix_data(new_data2)
    
    #run logistic regressions
    X_test_std,lr,y_test=log_reg(data,target,new_data2)
    
    #created confusion matrix
    conf_mat(X_test_std,lr,y_test)
    
#function to shuffle the data    
def shuffle_data(new_data):
    new_data = shuffle(new_data, random_state=0)
    return(new_data)

#plot Diagnosis N vs. O
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

    #or

    data_F['Diagnosis'].value_counts()
    sns.countplot(x='Diagnosis', data=data_F, palette="coolwarm")
    plt.show()
#function that converts to dummy values    
def dummy_new_table(X,data_F):
    dummy_Season=pd.get_dummies(data_F['Season'],prefix='Season')
    #print(dummy_Season)
    dummy_Childish_diseases=pd.get_dummies(data_F['Childish_diseases'],prefix='Childish_diseases')
    #print(dummy_Childish_diseases)
    dummy_Accident_trauma=pd.get_dummies(data_F['Accident_trauma'],prefix='Accident_trauma')
    #print(dummy_Accident_trauma)
    dummy_Surgical_intervention=pd.get_dummies(data_F['Surgical_intervention'],prefix='Surgical_intervention')
    #print(dummy_Surgical_intervention)

    new_data=pd.concat([X, dummy_Season,dummy_Childish_diseases,dummy_Accident_trauma,dummy_Surgical_intervention], axis=1)
    return(new_data)
  
#function for first correlation plot
def corr_fun(new_data):
    sns.heatmap(new_data.corr())
    corr = new_data.corr()
    return(corr)

#function for getting second correlation matrix
def new_corr_fun(new_data2):
    new_corr_matrix=new_data2.as_matrix()
    c=np.corrcoef(new_corr_matrix.transpose())
    print(np.shape(c))
    new_corr = np.around(c, decimals=2)
    print(new_corr)
    return(new_corr)

#function that plot new correlation matrix
def plotly_new_corr(new_corr):
    y = ['Diagnosis_code', 'Age', 'High_fevers', 'Alcohol_consumption','Smoking','Sitting','Season_-0.33','Season_0.33','Season_1.0','Childish_diseases_1','Accident_trauma_1','Surgical_intervention_1' ]
    x = ['Diagnosis_code', 'Age', 'High_fevers', 'Alcohol_consumption','Smoking','Sitting','Season_-0.33','Season_0.33','Season_1.0','Childish_diseases_1','Accident_trauma_1','Surgical_intervention_1']
    print(new_corr[0])
    colorscale = [[0, '#F5F5F5'], [1, '	#104E8B']]
    font_colors = ['#3c3636', '	#FCFCFC']
    fig = ff.create_annotated_heatmap(new_corr,x=x, y=y, colorscale=colorscale, font_colors=font_colors)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 6
    plot(fig, filename='correlation')
    
#function that represent the data as matrix and devides it on target variable and
#independent variables
def matrix_data(new_data2) :   
    data_F_matrix = new_data2.as_matrix()
    data=data_F_matrix[:,1:] 
    target=data_F_matrix[:,0]
    return (data, target)
    
#function of logistic regressions
def log_reg(data,target,new_data2):  
    X = data
    y = target
    print(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #print(X_train, X_test, y_train, y_test)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    LogisticRegression(penalty='l1')
    lr = LogisticRegression(penalty='l1', C=0.5)
    lr.fit(X_train_std, y_train)
    print('Training accuracy:', lr.score(X_train_std, y_train))
    print('Test accuracy:', lr.score(X_test_std, y_test))
    print(lr.intercept_)
    print(lr.coef_)

    lr_pred_prob=lr.predict_proba(X_test_std[0,:])
     
    #way to choose parameters 
    fig = plt.figure()
    ax = plt.subplot(111)
    colors = ['blue', 'green', 'red', 'cyan','maroon','magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue','gray', 'indigo', 'orange','orchid']
    weights, params = [], []
    for c in np.arange(-5, 6,  dtype=float):
        lr=LogisticRegression(penalty='l1', C=10**c, random_state=0)
      
        lr.fit(X_train_std,y_train)
        weights.append(lr.coef_[0])
        params.append(10**c)
    
    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]), colors):
        plt.plot(params, weights[:,column],label=new_data2.columns[column+1],color=color)

    plt.axhline(0, color='black', linestyle='--', linewidth=2)
    plt.xlim([10**(-3), 10**5])
    plt.ylabel('weight coefficient')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')
    ax.legend(loc='upper center',
    bbox_to_anchor=(1.38, 1.03),
    ncol=1, fancybox=True)
    plt.show()
    return(X_test_std,lr,y_test)

#function for confusion matrix
def conf_mat(X_test_std,lr,y_test):
    lr_pred_prob=lr.predict_proba(X_test_std)
    
    print(np.around(lr_pred_prob, decimals=3))
    
    y_pred = lr.predict(X_test_std)
    print(y_pred)
    print(y_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Greens, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

#function for sigmoid
def sigmoid(z):
   return (1.0 / (1.0 + np.exp(-z)))

#function for plotting sigmoid
def sigm_plot(z,phi_z):
    plt.plot(z, phi_z,'#3A5FCD')
    plt.axvline(0.0, linewidth=1,color='k')
    plt.axhspan(-0.1, 1.1, facecolor='1.0', alpha=1.0, ls='dotted')
    plt.axhline(y=0.5, ls='dotted', color='k')
    plt.axhline(y=0.0, ls='dotted',color='k')
    plt.axhline(y=1.0, ls='dotted', color='k')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.show()

def prep_data(data_F):
    #converted Diagnosis values 'Yes' and 'No' to 0 and 1, dropped Diagnosis column
    LE_Diagnosis = LabelEncoder()
    data_F["Diagnosis_code"] = LE_Diagnosis.fit_transform(data_F["Diagnosis"])
    data_F[["Diagnosis", "Diagnosis_code"]].head()
    data_F=data_F.drop(['Diagnosis'],axis=1)
    print(data_F.head())

    #placed Diagnosis_code to the first column
    D_C = data_F['Diagnosis_code']
    data_F.drop(labels=['Diagnosis_code'], axis=1,inplace = True)
    data_F.insert(0, 'Diagnosis_code', D_C)
    #print(data_F.head())
    
    #placed Age to the second column
    A=data_F['Age']
    data_F.drop(labels=['Age'], axis=1,inplace = True)
    data_F.insert(1, 'Age', A)
    #print(data_F.head())
    return(data_F)
main()








