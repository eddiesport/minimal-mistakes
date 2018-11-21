# -*- coding: utf-8 -*-
"""
More advanced FPL classification model
Scaling our data, putting it in a pipeline.

Created on Sun Nov 18 13:07:35 2018

@author: Eddie
"""

#%%#Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

#%%#Import our csv data file as a DF

fpl_data = pd.read_csv('https://eddiesport.github.io/projects/FPL_final.csv', \
                       header=0, index_col=0, encoding='cp1252')

print(fpl_data.head())
print(fpl_data.info())

#%%#Using Label Encoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
fpl_data.Position = le.fit_transform(fpl_data.Position)
class_names = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(class_names, values))
print(dictionary)

#%%#Create feature (X) and target (y) arrays
feature_cols = ['Goals Scored', 'Assists', 'Clean Sheets', 'Goals Conceded', \
                'Own Goals', 'Penalties Saved', 'Penalties Missed', \
                'Yellow Cards', 'Red Cards', 'Saves', 'Bonus Points', \
                'Bonus Point System Score', 'Influence', 'Creativity', \
                'Threat', 'ICT Index']

X = fpl_data.loc[:, feature_cols]
print(X.shape)

y = fpl_data.Position
print(y.shape)

print(y.value_counts())

#%%#Create a show a correlation matrix of the feature columns
corr_mat = X.corr()

sns.heatmap(corr_mat)
plt.title('Feature Columns Correlation Heatmap')
plt.show()
plt.clf()

#%%#Quick look at the data

print(fpl_data.describe())

"""Can see a large variation in range, eg Assists 0-18, to Threat 0-2355.
Lets scale and centre the data using Standardization"""

#%%#Import a Scaler and a Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#%%#Set up the pipeline steps and create the pipeline object
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps)

#Define parameter grid for K neighbors search
k_range = np.arange(1, 21)
parameters = {'knn__n_neighbors': k_range}

print(parameters)

#%%#Create training and test (hold out) sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, \
                                                    random_state=42, stratify=y)

#%%# Instantiate GridSearch object, cv=3 default arg here
fpl_model = GridSearchCV(pipeline, parameters, cv=5)

#Fit pipeline to training set
fpl_model.fit(X_train, y_train)

#Predict labels of the test data
y_pred = fpl_model.predict(X_test)

#%%#Compute and print metrics
from sklearn.metrics import classification_report, confusion_matrix

#Print accuracy, confusion matrix and the classification report
print('Accuracy: {}'.format(fpl_model.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(fpl_model.best_params_))

#%%##%%#Lets visualize a confusion matrix

#Function to visualize the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


#%%# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

"""We have class imbalance(far more MF, DF than FW, GK), so lets normalize our confusion matrix"""

#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#%%#
"""Can compute accuracy from confusion matrix, sum of diagonal (tp+tn)/sum all or 105/133
Can compute precision tp/(tp+fp), eg for DF its 37/44, for MF its 40/53, etc
Can compute recall tp/(tp+fn), eg for DF recall is 37/48, for MF its 40/55, etc
F1 Score is 2*(precision*recall)/(precision+recall) eg for DF 2*(0.84*.077)/(0.84+0.77)
High Precision - Not many real defenders predicted as MF, FW, GK, etc
High Recall - Most FW predicted correctly as forwards"""

#%%#Use model on full dataset
model_result = fpl_model.predict(X)

#%%#Model result is a list of class_names
#print(model_result)

#%%#
print('Accuracy: {}'.format(fpl_model.score(X, y)))
print(classification_report(y, model_result))

#%%#Lets visualize a confusion matrix

#%%# Compute confusion matrix on whole data set
cnf_matrix = confusion_matrix(y, model_result)

#Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

"""We have class imbalance(far more MF, DF than FW, GK), so lets normalize our confusion matrix"""

#Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#%%#Add model result to full dataset
fpl_data['Results'] = model_result
print(fpl_data.head())

#%%#Selection of players
fpl_wrong = fpl_data.loc[fpl_data['Position'] != fpl_data['Results']]
#print(fpl_wrong)

print(fpl_wrong[(fpl_wrong['Position'] == 3) & (fpl_wrong['Results'] == 1)])

"""We find 8 players the model thinks should be strikers but are listed as midfielders.
Are these potentially good picks for future fantasy teams?
Salah, Richarlson, Zaha, James McArthur, etc"""