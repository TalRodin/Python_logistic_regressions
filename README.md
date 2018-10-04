# Python_logistic_regressions
(in process) Prediction of fertility of women based on different features, using logistic regression algorithm. Working on Medium and separate report of analyses
![image](https://user-images.githubusercontent.com/32145723/46493872-6cb16680-c7df-11e8-8286-30e10e2652e1.png)
There are four types of Machine Learning algorithms: Supervised, Unsupervised, Semi-Supervised, and Reinforcement. Supervised Machine Learning algorithms are classification and regression. Classification problems is when the dependent variable is categorical such as gender(male or female), color (black or white), nationality, employment status, etc. Another words, the output is binary. But to say “yes” or “no”, Schrödinger’s cat alive or dead is not enough in this case. We would like to know the chances of the outcomes, having input variables. 

For our example of Logistic Regressions, we will use the dataset from https://archive.ics.uci.edu/ml/datasets/Fertility to predict the Fertility analyses, which classified as normal (N) and altered (O). The diagnosis is our target or outcome. The input variables:

Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall (-1, -0.33, 0.33, 1) — nominal categorical data
Age at the time of analysis. 18–36 (0, 1) — ordinal categorical data
Childish diseases (ie , chicken pox, measles, mumps, polio). 1) yes, 2) no (0, 1) — nominal categorical data
Accident or serious trauma. 1) yes, 2) no (0, 1) — nominal categorical data
Surgical intervention. 1) yes, 2) no. (0, 1) — nominal categorical data
High fevers in the last year. 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1) — ordinal categorical data
Frequency of alcohol consumption. 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1) — ordinal categorical data
Smoking habit. 1) never, 2) occasional 3) daily. (-1, 0, 1) — ordinal categorical data
A Number of hours spent sitting per day 1–16 (0, 1) — ordinal categorical data

