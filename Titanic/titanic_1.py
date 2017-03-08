from __future__ import division
import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


# read the titanic data set 
titanic = pandas.read_csv("train.csv")

# describe 
print(titanic.describe())

# print age
print(titanic["Age"])

median = titanic["Age"].median()

print("replacing NaNs with median of Age", median)

titanic["Age"] = titanic["Age"].fillna(median)


# print age with corrected data
print(titanic["Age"])

# convert Sex to numeric values, male=0, female=1

print(titanic["Sex"].unique())

# assign all males to 0
titanic.loc[titanic["Sex"]=="male","Sex"]=0

# assign all females to 1
titanic.loc[titanic["Sex"]=="female","Sex"]=1

print(titanic["Sex"])
	

print(titanic["Embarked"].unique())

titanic["Embarked"]=titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"]=="S","Embarked"] = 0
titanic.loc[titanic["Embarked"]=="C","Embarked"] = 1
titanic.loc[titanic["Embarked"]=="Q","Embarked"] = 2

print(titanic["Embarked"].unique())

# describe 
print(titanic.describe())

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg=LinearRegression()

kf=KFold(titanic.shape[0],n_folds=3,random_state=1)
print kf

predictions=[]

for train,test in kf:
	train_predictors=(titanic[predictors].iloc[train,:])
	train_target=titanic["Survived"].iloc[train]
	alg.fit(train_predictors,train_target)

	test_predictions = alg.predict(titanic[predictors].iloc[test,:])
	predictions.append(test_predictions)

print titanic.ndim
print titanic.ftypes

predictions = np.concatenate(predictions,axis=0)

predictions[predictions > 0.5] =1
predictions[predictions <= 0.5] =0
match =0
n=titanic["Survived"].size
print n
for i in range(0 , n):
	if (predictions[i] == titanic["Survived"][i]):
		match = match+1
print match
print predictions.size
print "Accuracy % =",match / predictions.size * 100


# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


titanic_test = pandas.read_csv("test.csv")

# fix AGE
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

# fix SEX
titanic_test.loc[titanic_test["Sex"]=="male","Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female","Sex"]=1

# fix Embarked
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"]=="S","Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C","Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q","Embarked"]=2

#fix Fare
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle_1.csv", index=False)