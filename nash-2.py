# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 23:08:44 2022

@author: Ibrahim2
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


data = pd.read_csv('NASH-2-Red-All-num.csv')
print(data.head())
data['FamilyHistoryQ'].unique()
data['FamilyHistoryQ']-=1
#print(data.columns)
new_data = data.iloc[:,0:10]
#print(new_data.head())

data['sexQ'].unique()
data['sexQ']-=1
#print(data['Group2'].unique().tolist())

y = data['Group2'].values.tolist()
X = data.drop('Group2', axis =1).values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#y_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)

#clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
#
#clf = OneVsRestClassifier(RandomForestClassifier(max_depth=4, random_state=4)).fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#
#
#y = new_data['Group2'].values.tolist()
#X = new_data.drop('Group2', axis =1).values
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y)
#
#
#scaler = MinMaxScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
#
#y_pred = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train).predict(X_test)

#clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
#y_pred = clf.predict(X_test)

#clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)

clf = OneVsRestClassifier(RandomForestClassifier(max_depth=4, random_state=0)).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.linear_model import LogisticRegression
def LROptimize(X,Y,opt):    
    clf = LogisticRegression(random_state=42)
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [300,200,150,100, 10, 1.0, 0.1]
            
    grid_values = {'solver': solvers, 'penalty': penalty, 'C':c_values}
    #scoring_metrics = ['f1','balanced_accuracy']
    grid_model = GridSearchCV(clf, param_grid = grid_values,scoring = opt, cv=5)
    grid_model.fit(X, Y)
    print("**********************************************")
    print("CV BA:",grid_model.cv_results_['mean_test_score'])
    print("**********************************************")
    print(grid_model.best_estimator_)
    return grid_model.best_estimator_

from sklearn.model_selection import GridSearchCV
def RFOptimize(X,Y,opt):
    clf = RandomForestClassifier(random_state=42)
   
    n_estimators = [50,100,200,300]
    max_depth = [3,4,5]
    min_samples_split = [3, 5, 7]
    min_samples_leaf = [1, 2, 3] 
            
    grid_values = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
    #scoring_metrics = ['f1','balanced_accuracy']
    grid_model = GridSearchCV(clf, param_grid = grid_values,scoring = opt, cv=3)
    grid_model.fit(X, Y)
    print("**********************************************")
    print("CV BA:",grid_model.cv_results_['mean_test_score'])
    print("**********************************************")
    print(grid_model.best_estimator_)
    return grid_model.best_estimator_


from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot


#opt_lr = LROptimize(X_train,y_train,"precision_weighted") 
#opt_rf = RFOptimize(X_train,y_train,"precision_weighted") 


#opt_lr.
clf_lr = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)
#clf_lr = OneVsRestClassifier(opt_rf).fit(X_train, y_train)

y_pred = clf_lr.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




opt_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=4, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
            oob_score=False, random_state=42, verbose=0, warm_start=False)

opt_fin_rf = OneVsRestClassifier(opt_rf).fit(X_train, y_train)
y_pred = opt_fin_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets

# Binarize the output
y = label_binarize(y, classes=[1, 2, 3,4])
n_classes = y.shape[1]

#classifier = OneVsRestClassifier(
#    svm.SVC(kernel="linear", probability=True, random_state=0))

y_score = opt_fin_rf.predict_proba(X_test)
y_score = np.array(y_score)
data = list(y_test)
y_test = np.array(data)


# precision recall curve
precision = dict()
recall = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()


# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#
#
## First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
## Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#
## Finally average it and compute AUC
#mean_tpr /= n_classes
#
#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
## Plot all ROC curves
#plt.figure()
#plt.plot(
#    fpr["micro"],
#    tpr["micro"],
#    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#    color="deeppink",
#    linestyle=":",
#    linewidth=4,
#)
#
#plt.plot(
#    fpr["macro"],
#    tpr["macro"],
#    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
#    color="navy",
#    linestyle=":",
#    linewidth=4,
#)
#
#colors = cycle(["aqua", "darkorange", "cornflowerblue"])
#for i, color in zip(range(n_classes), colors):
#    plt.plot(
#        fpr[i],
#        tpr[i],
#        color=color,
#        lw=lw,
#        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#    )
#
#plt.plot([0, 1], [0, 1], "k--", lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("Some extension of Receiver operating characteristic to multiclass")
#plt.legend(loc="lower right")
#plt.show()
#






#lr_probs = opt_fin_rf.predict_proba(X_test)
#lr_probs = lr_probs[:, 1]
#lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
#no_skill = len(y_test[y_test==1]) / len(y_test)
#pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
#pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
## axis labels
#pyplot.xlabel('Recall')
#pyplot.ylabel('Precision')
## show the legend
#pyplot.legend()
## show the plot
#pyplot.show()