# Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')
#Creat Matrix of Features (array of features)
#iloc is a function of pandas, that will take the indexes 
#of columns that we want to extract from the dataset
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data, one common way is:
# to take the mean of available data in that column
#to do that we use scikit-learn library and import
# imputer class
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values =np.nan, strategy = 'mean')
# to fit this imputer object into our matrix of x:
imputer.fit(x[:,1:3])
#now replace the missing data of the matrix x by the mean of column
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encoding categorical variables, meaning:
#encoding variables that are text into numbers: Country & Purchase

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# now creat the first object of labelencoder class
labelencoder_X = LabelEncoder()
#to apply lableencoder object to our column
x[:,0] = labelencoder_X.fit_transform(x[:,0])
#all these return the 1st column country of matrix x encoded
#so we have the encoded values of these countries
# however there might be a confusion for machine-learning model
#our program might think that spain has higher value than germany
#but these numbers are not here to indicate priority or greatness
#therefore  to prevent this we use dummy variables, means:
#instead of having one column, we will have more columns.
# 3 columns equal to the number of categories
# we import onehotencoderclass, and creat its object
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)
#onehotencoder = OneHotEncoder(categories = np.array([0,1,2]))
#x = onehotencoder.fit_transform(x).toarray()

# for the purchase label, we need to use only the labelencoder
#because its a dependent variable and the machine learning model
#knows that its a category and there is no order between the two

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# now we need to split the data set into training and test set
#library that does that for us
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Feature Scaling: when independent variables are not the same order
#Euclidean distance (most of ML models) will be dominated by one Var.
#Therefore we change the scale independent variables so that
#they will have the same range
# 1 way is called standardization: X_stand = (X-X_mean)/stand.deriv(x)
#2nd way, normalization : X_norm = (X-min(X))/(max(X)-min(X))
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
#you have to fit the object to the training set and then transform it
#but for the test set we just transform there is no need to fit again
#because it has been fitted before in the training set
x_test =sc_x.transform(x_test)

# do we need to fit and transport the dummy variables?
# for this case we should not because then we lose the interpretation
#right now we have 0 & 1 which makes sence for countries, if we scale
#then we will lose sence of the country, however we just do it




