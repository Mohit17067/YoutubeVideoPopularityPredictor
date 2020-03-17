import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from xgboost import plot_importance
import seaborn as sns
from scipy import stats
from sklearn.model_selection import GridSearchCV
import sklearn.svm as sk



def GridSearch(X,Y,model):
	penalisation_constant=np.array([1,2,3,4,5,6,7,8,9,10])
	param_grid = dict(max_depth=penalisation_constant)
	grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=5)
	grid.fit(X,Y)
	return grid.best_estimator_.max_depth


features = ['duration','description_length','dislikeCount','commentCount','likeCount','channel_ViewCount','channel_subscriberCount','channel_videoCount','video_title_length','No_of_tags','social_links']

df_initial = pd.read_csv("data.csv")

#print(df.head(5))

df = df_initial.loc[:, ['duration','description_length','dislikeCount','commentCount','likeCount','channel_ViewCount','channel_subscriberCount','channel_videoCount','video_title_length','No_of_tags','social_links','viewCount']]

df = df.sample(frac=1)


X = df.drop(['viewCount'], axis =1)
# df.plot.hist(subplots=True, layout=(6,2))
# plt.show()

X = np.array(X)
Y = df[df.columns[-1]]
Y = np.array(Y)

X = np.array(X, dtype=np.float64)
Y = np.array(Y, dtype=np.float64)

df = np.array(df)

X = np.array(X, dtype=np.float64)
Y = np.array(Y, dtype=np.float64)
X = X[:100000, :]
Y = Y[:100000]
mean = np.mean(X,axis=0)
std = np.std(X,axis=0)
print(std)
meanLab = np.mean(Y,axis=0)
stdLab = np.std(Y,axis=0)


print(stdLab)
print(mean)
X = (X-mean)/std
Y = (Y-meanLab)/stdLab


delRows = []

zscore = np.abs(stats.zscore(df))

for i in range(len(X)):
	for j in range(len(X[0])):
		if(abs(X[i,j])>4):
			delRows.append(i)
			break

X_new = np.delete(X,delRows,axis=0)
Y_new = np.delete(Y,delRows,axis=0)

print(X_new.shape)
print(Y_new.shape)


delRows = []

for i in range(len(Y_new)):
	if(abs(Y_new[i])>10):
		delRows.append(i)

X_new = np.delete(X_new,delRows,axis=0)
Y_new = np.delete(Y_new,delRows,axis=0)



print(X_new.shape)
print(Y_new.shape)

print(X.shape)
print(Y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(X_new, Y_new, test_size = 1/5, random_state = 0)
# print(xTrain.shape)
# print(xTest.shape)
# print(yTrain.shape)
# print(yTest.shape)

linearRegressor = LinearRegression()

linearRegressor.fit(xTrain, yTrain)
yPrediction = linearRegressor.predict(xTest)

print(linearRegressor.coef_)


# clf = SVR(cache_size=5000)
# clf.fit(xTrain[:150000], yTrain[:150000])
# print("SVR" + str(clf.score(xTrain[:150000], yTrain[:150000])))
# print("SVR" + str(clf.score(xTest[:30000], yTest[:30000])))
# #plot.scatter(xTrain, yTrain, color = 'red')	

# plot.plot(xTrain, linearRegressor.predict(xTrain), color = 'blue')
# plot.show()
print("Linear Regression:")
print()
print("Accuracy on the Training Data:" + " " + str(linearRegressor.score(xTrain,yTrain)))
print("Accuracy on the Test Data:" + " "  + str(linearRegressor.score(xTest,yTest)))


# depth = GridSearch(X,Y,RandomForestRegressor())
# print(depth)
regr = RandomForestRegressor(max_depth = 3)
regr.fit(xTrain, yTrain)

print("Random Forest, max-depth=3:")
print()
print("Accuracy on the Training Data:" + " " + str(regr.score(xTrain,yTrain)))
print("Accuracy on the Test 5Data:" + " "  + str(regr.score(xTest,yTest)))


regr = RandomForestRegressor(max_depth = 4)
regr.fit(xTrain, yTrain)

print("Random Forest, max-depth=4:")
print()
print("Accuracy on the Training Data:" + " " + str(regr.score(xTrain,yTrain)))
print("Accuracy on the Test 5Data:" + " "  + str(regr.score(xTest,yTest)))


regr = RandomForestRegressor(max_depth = 5)
regr.fit(xTrain, yTrain)

print("Random Forest, max-depth=5:")
print()
print("Accuracy on the Training Data:" + " " + str(regr.score(xTrain,yTrain)))
print("Accuracy on the Test 5Data:" + " "  + str(regr.score(xTest,yTest)))


regr = RandomForestRegressor(max_depth = 6)
regr.fit(xTrain, yTrain)

print("Random Forest, max-depth=6:")
print()
print("Accuracy on the Training Data:" + " " + str(regr.score(xTrain,yTrain)))
print("Accuracy on the Test 5Data:" + " "  + str(regr.score(xTest,yTest)))

regr = RandomForestRegressor(max_depth = 7)
regr.fit(xTrain, yTrain)

print("Random Forest, max-depth=7:")
print()
print("Accuracy on the Training Data:" + " " + str(regr.score(xTrain,yTrain)))
print("Accuracy on the Test 5Data:" + " "  + str(regr.score(xTest,yTest)))


regr = RandomForestRegressor(max_depth = 8)
regr.fit(xTrain, yTrain)

print("Random Forest, max-depth=8:")
print()
print("Accuracy on the Training Data:" + " " + str(regr.score(xTrain,yTrain)))
print("Accuracy on the Test 5Data:" + " "  + str(regr.score(xTest,yTest)))


importances = regr.feature_importances_

# print()
regressor = DecisionTreeRegressor()
regressor.fit(xTrain, yTrain)
# importances2 = regressor.feature_importances_

print("Decision Tree:")
print()
print("Accuracy on the Training Data:" + " " + str(regressor.score(xTrain,yTrain)))
print("Accuracy on the Test 5Data:" + " "  + str(regressor.score(xTest,yTest)))

svm_regressor = sk.SVR(kernel='rbf')
svm_regressor.fit(xTrain,yTrain)

print("SVR:")
print()
print("Accuracy on the Training Set:" + " " + str(svm_regressor.score(xTrain,yTrain)))
print("Accuracy on the Test Set:"+" " + str(svm_regressor.score(xTest,yTest)))




indices=np.argsort(importances)
print(indices)

plt.subplot(5,1,1)
plt.scatter(X[:,5],Y,color='blue')
plt.xlabel("Channel View Count")

plt.subplot(5,1,2)
plt.scatter(X[:,6],Y,color='brown')
plt.xlabel("Channel Subscriber Count")
# plt.subplot(4,1,3)
# plt.scatter(X_new[:,5],Y_new,color='red')

# plt.subplot(4,1,4)
# plt.scatter(X_new[:,6],Y_new,color='green')

plt.subplot(5,1,3)
plt.scatter(X[:,2],Y,color='red')
plt.xlabel("Dislike Count")

plt.subplot(5,1,4)
plt.scatter(X[:,3],Y,color='green')
plt.xlabel("Comment Count")

plt.subplot(5,1,5)
plt.scatter(X[:,4],Y,color='pink')
plt.xlabel("Like Count")
# plt.barh(range(len(indices)),importances[indices])
# plt.yticks(range(len(indices)),[features[i] for i in indices])
plt.show()


plt.subplot(5,1,1)
plt.scatter(X_new[:,5],Y_new,color='blue')
plt.xlabel("Channel View Count")

plt.subplot(5,1,2)
plt.scatter(X_new[:,6],Y_new,color='brown')
plt.xlabel("Channel Subscriber Count")


plt.subplot(5,1,3)
plt.scatter(X_new[:,2],Y_new,color='red')
plt.xlabel("Dislike Count")

plt.subplot(5,1,4)
plt.scatter(X_new[:,3],Y_new,color='green')
plt.xlabel("Comment Count")

plt.subplot(5,1,5)
plt.scatter(X_new[:,4],Y_new,color='pink')
plt.xlabel("Like Count")

plt.show()

# plt.show()
# indices=np.argsort(importances)
# print(indices)
# plt.barh(range(len(indices)),importances[indices])
# plt.yticks(range(len(indices)),[features[i] for i in indices])
# plt.show()
