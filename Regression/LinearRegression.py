import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# load the dataset
dataset = np.loadtxt("watermelon_dataset.txt", delimiter="\t")
DensityAndSugar = dataset[:, 1:3]
Label = dataset[:, 3]
print("Density and sugar: ")
print(DensityAndSugar)
print("Label:")
print(Label)

# draw the plot of the dataset
f1 = plt.figure(1)
# label of the axis
plt.xlabel('Density')
plt.ylabel('Sugar')
# title
plt.title('Watermelon')
# scatter point
plt.scatter(DensityAndSugar[Label == 0, 0], DensityAndSugar[Label == 0, 1], s=50, marker='o', color='k', label='bad')
plt.scatter(DensityAndSugar[Label == 1, 0], DensityAndSugar[Label == 1, 1], s=50, marker='o', color='g', label='good')
# the location of the legend
plt.legend(loc='upper left')

# split the train set and test set
xtrain, xtest, ytain, ytest=model_selection.train_test_split(DensityAndSugar, Label, test_size=0.5, random_state=0)
# logistic regression model
model=LogisticRegression()
# train
model.fit(xtrain, ytain)

# test
ypred=model.predict(xtest)

score=cross_val_score(model, DensityAndSugar, Label, cv=5)

# result of the test
# confusion matrix
print("ytest: ")
print(ytest)
print("ypred: ")
print(ypred)
print("confusion matrix: ")
print(metrics.confusion_matrix(ytest, ypred))
# classifier
print("classifier: ")
print(metrics.classification_report(ytest, ypred))
print("score: ")
print(score)
plt.show()