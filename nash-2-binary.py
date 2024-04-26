# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:37:51 2022

@author: Ibrahim2
"""

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

    
import numpy as np
import pandas as pd

from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import matthews_corrcoef
from pandas import DataFrame
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
simplefilter(action='ignore', category=RuntimeWarning)

def SVCLinearFeatures(all_data,x,y, fname, fprint=False,stp=10, fno=10,opt="f1"):
    
    svc = SVC(kernel = 'linear', degree =3, gamma='auto', C=1,probability=True, )
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), scoring="f1")
    rfecv = rfecv.fit(x, y)
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features")
#    plt.ylabel("Cross validation score (nb of correct classifications)")    
    plt.ylabel("Cross validation scores")

    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    print("Lin. SVC Len.: ",len(rfecv.ranking_))
#    print(rfecv.get_support(indices=True))
    sel = rfecv.get_support(indices=True)
    features_sel = all_data.iloc[:,sel] 
    print("feature_sel:", features_sel.columns)
    print("features_sel",features_sel.shape)
#    if fprint:
#    ### Code to write the selected feature names into a file  ****
#        sel = rfecv.get_support(indices=True)
#        features_sel = all_data.iloc[:,sel] 
#        sel_feat = [col for col in features_sel.columns]
#        print("features_sel:",features_sel.columns)
#        set_name = "-".join(fname.split("-")[:2])
#        featFile =p+set_name + "-feat"+str(len(sel_feat))+".csv"
#        f= open(featFile,"w")
#        for i in sel_feat:
#            f.write(i+'\n')
#        f.close()    
    ##### Code to write the selected features data into a file, file name is split from the original file name *****    
#        new_ds =  pd.concat([features_sel,all_data['Bmode']], sort=False)
#        set_name = "-".join(fname.split("-")[:2])
#        feat_dataFile = p+set_name + "-SVCdata"+str(len(sel_feat))+".csv"
#        new_ds.to_csv(feat_dataFile, sep=',',index=False)    
    
    return rfecv

def SVCOptimize(X,Y,opt):
    from sklearn.model_selection import GridSearchCV
    clf = SVC(probability=True, random_state=42)
    #kernels = ['poly','rbf','sigmoid','precomputed','callable']
    #gammas = ['auto','scale', 0.01,0.1,1,10,100,500]
    Cs = [1.0,2,5,10,15,80,100,150,200]
    #Cs = [100,150, 170, 200,300]
#    degrees = [1,2,4]
    kernls = ['rbf','poly']
    degrees = [1,2,3,4,5]

    grid_values = {'kernel': kernls, 'degree': degrees, 'C':Cs}
    #scoring_metrics = ['f1','balanced_accuracy']
    grid_model = GridSearchCV(clf, param_grid = grid_values,scoring = opt)
    grid_model.fit(X, Y)
    print("**********************************************")
    print("CV BA:",grid_model.cv_results_['mean_test_score'])
    print("**********************************************")
    print(grid_model.best_estimator_)
    return grid_model.best_estimator_

def RFOptimize(X,Y,opt):
    clf = RandomForestClassifier(random_state=42)
   
    n_estimators = [20,50,100]
    max_depth = [3,4,5,8]
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

def Prec_Recall_curve(prec, rec):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    
    from matplotlib import pyplot
    
    lr_probs = opt_rf1.predict_proba(xtst_val)
    lr_probs = lr_probs[:, 1]
    lr_precision, lr_recall, _ = precision_recall_curve(ytst_val, lr_probs)
    #lr_f1, lr_auc = f1_score(ytst_val, y_pred_val), auc(lr_recall, lr_precision)
    # summarize scores
    #print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(ytst_val[ytst_val==1]) / len(ytst_val)
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
def Normalize(x):
    scalar = MinMaxScaler()
    x_normal = scalar.fit_transform(x)
    return x_normal

def Calculate_Fscore(array, labels):
    if len(array) != len(labels):
        print('Error. inconsistent data shape with sample number')
        return 0

    array_po = []
    array_ne = []
    for i in range(len(labels)):
        if labels[i] == 1:
            array_po.append(array[i])
        else:
            array_ne.append(array[i])

    mean_po = sum(array_po) / len(array_po)
    mean_ne = sum(array_ne) / len(array_ne)
    mean = sum(array) / len(array)


    score_1 = ((mean_po - mean) ** 2 + (mean_ne - mean) ** 2)
    score_2 = sum([(i-mean_po) ** 2 for i in array_po]) / (len(array_po) - 1)
    score_3 = sum([(i-mean_ne) ** 2 for i in array_ne]) / (len(array_ne) - 1)
    f_score = score_1 / (score_2 + score_3)
    return f_score

def Fscore(encodings,features, labels):
          
    data = encodings.astype(float)
    shape = data.shape

    e = ''
    if shape[0] < 5 or shape[1] < 2:
        return 0, e

    dataShape = data.shape
    print(dataShape)

    print(dataShape[1])
    print(len(features))
    
    if dataShape[1] != len(features):
        print('Error: inconsistent data shape with feature number.')
        return 0, 'Error: inconsistent data shape with feature number.'
    if dataShape[0] != len(labels):
        print('Error: inconsistent data shape with sample number.')
        return 0, 'Error: inconsistent data shape with sample number.'

    myFea = {}
    for i in range(len(features)):
        array = list(data[:, i])
        try:
            myFea[features[i]] = Calculate_Fscore(array, labels)
        except (ValueError, RuntimeWarning) as e:
            pass
        
    res = []
    res.append(['feature', 'F-score'])
    for key in sorted(myFea.items(), key=lambda item: item[1], reverse=True):
        res.append([key[0], '{0:.3f}'.format(myFea[key[0]])])
    return res

def FScoreFeatures(x,y, fname,score_cutof=0,noFeat=0,fprint=False):

    from operator import itemgetter

    features = list(x.columns)
    x_norm = Normalize(x)
    x_norm_df = DataFrame(x_norm)
    x_norm_df.columns = features
    
    f_Scores = Fscore(x_norm, features,y.values)

    score = [float(s[1]) for s in f_Scores[1:]]
    score_avg = np.average(score)
    max_score = np.max(score)
#    print(sorted(score))
    sorted_f_Scores = sorted(f_Scores[1:], key=itemgetter(1),reverse=True)
#    print(sorted_f_Scores)
    print("Scores Len.: %s - Avg: %s - Max:%s - >Avg.:%s, >0 %s" %(len(score),score_avg,max_score,sum(i > score_avg for i in score),len([i>0 for i in score])))   

    final_sel = sorted_f_Scores
    if(score_cutof>0):
        final_sel = [f for f in sorted_f_Scores if float(f[1]) > score_cutof ] # Select only features over a cutoff of importance
        print("FScore Selected:%s - CutOf:%s" %(len(final_sel),score_cutof))

    ### Retreive specific number or percent of features
    if (noFeat > 1):
        cutPer = noFeat/len(sorted_f_Scores)
        final_sel = [f for f in sorted_f_Scores[:noFeat]] 
        print("FScore Selected:%s - Num:%s, Percent:%s" %(len(final_sel),noFeat,cutPer))

    elif(noFeat <= 1 and noFeat > 0):
        cutNo = int(noFeat*len(sorted_f_Scores))
        final_sel = [f for f in sorted_f_Scores[:cutNo]] 
        print("FScore Selected:%s - Num:%s, Percent:%s" %(len(final_sel),cutNo,noFeat))
#    print(final_sel)

    feat_df = DataFrame(final_sel)
    feat_df.columns = ['feature', 'Importance']  # dataframe for feature names and importance
 
    new_ds = x_norm_df[feat_df['feature']]
    print("New FScore DS: ", new_ds.shape)

    # Code to write 2 files for feature names and data, file names are extracted from original name
    if fprint:
        set_name = "-".join(fname.split("-")[:2])
        featFile = p+set_name + "-feat"+str(len(final_sel))+".csv"
        feat_dataFile = p+set_name + "-data"+str(len(final_sel))+".csv"
        new_ds.to_csv(feat_dataFile, sep=',',index=False)
        feat_df.to_csv(featFile, sep=',',index=False)
    
    return new_ds

def Draw_PCA(X_train_std,y_train):
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    #
    ## Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
#    print('Matrix W:\n', w)
    X_train_std[0].dot(w)
    X_train_pca = X_train_std.dot(w)
    print(len(X_train_pca))
#    print(X_train_pca)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l, 0], 
                    X_train_pca[y_train==l, 1], 
                    c=c, label=l, marker=m) 
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower right')
    plt.show()
 
####################################
def crossValidOneNormalized(x,y,rs=0,tSize=0.2):
    "Returns one train/test split"
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=tSize, random_state=rs,stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test 

def Plot_avg_ROC(X,y,model):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)    

    #print('AVG. AUC (Compounds):',1) #+ ' {0:.2f}'.format(auc)
    
    plt.figure(figsize=(5, 5))
#    model = SVC(kernel = 'poly', degree =3, C=1,probability=True)#SVC(kernel='rbf',probability=True)
    
    folds = 5
    avgAuc, avgScore = 0,0
    
    #cv = KFold(n=len(y), n_folds=10, random_state=42, shuffle=False)
    #cv.
    for i in range(folds):
    #    print("Loop1: ",i)
    
        xtr, xtst, ytr, ytst = crossValidOneNormalized(X,y,i*5)
    #    print("Train Index: ", train_index, "\n")
    #    print("Test Index: ", test_index)
    
    #    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        model.fit(xtr, ytr)
  
    #    scores.append(best_svr.score(X_test, y_test))
        s1 = model.score(xtst, ytst)
    #    print("S1:",s1)
    #    print("S2:", s2)
        avgScore +=s1
    
        y_score = model.predict_proba(xtst)
        predict = y_score[:, 1]
        auc = roc_auc_score(ytst, predict)
        avgAuc += auc
        print("Comp auc:%2.3f -- Score:%2.3f " %(auc,s1))
        fpr, tpr, _ = roc_curve(ytst, y_score[:, 1])
    
        plt.plot(fpr, tpr, 'b', alpha=0.15, label=auc)
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)  
    
    avgAuc = avgAuc/folds
    avgScore = avgScore/folds
    
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std    
   
    plt.plot(base_fpr, mean_tprs, 'b')
    
    #plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,label='labelllllllllllll')
    plt.plot([0, 1], [0, 1],'r--')
    #print('AVG. AUC (Compounds):',5) 
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
    print("%d-Fold Cross Validation" % folds)
    print("AVG. AUC : %3.2f" %(avgAuc))
    print("AVG. Score: %3.2f" %(avgScore))

    
    
    
    
    
    
    
    
    
    
    
    
    
####################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import time
start_time = time.time()
print(time.strftime('%X %x'))

df = pd.read_csv('E:\\Banha\\Dr Noha project\\NASH-2-Red-All-num-zero-bin-Nob.csv')

trgName = "Group2" 
yP = df[trgName]
xP = df.drop(trgName,axis=1)
x_norm_all = Normalize(xP)
xtr_val, xtst_val, ytr_val, ytst_val = train_test_split(x_norm_all, yP, test_size=0.2,random_state=42, stratify=yP)

Draw_PCA(xtr_val,ytr_val)
#rfecv = SVCLinearFeatures(df,xtr_val, ytr_val,"sel1.csv",False,3,5)
#xtr_val = rfecv.transform(xtr_val)
#xtst_val = rfecv.transform(xtst_val) # Validation set minimized features
print("Lin. SVC: ",xtr_val.shape)

#opt_lr = LROptimize(X_train,y_train,"precision_weighted") 
opt_rf = RFOptimize(xtr_val,ytr_val,"f1") 

#opt_rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=8, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=5,
#            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
#            oob_score=False, random_state=42, verbose=0, warm_start=False)
#opt_rf.fit(xtr_val,ytr_val)

#opt_lr.
#clf_lr = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)
#clf_lr = OneVsRestClassifier(opt_rf).fit(X_train, y_train)
from sklearn.metrics import classification_report

y_pred = opt_rf.predict(xtst_val)
#y_pred = clf_lr.predict(X_test)
print(confusion_matrix(ytst_val, y_pred))
print(classification_report(ytst_val, y_pred))


y_pred = opt_rf.predict(xtst_val)
#y_pred = clf_lr.predict(X_test)
print(confusion_matrix(ytst_val, y_pred))
print(classification_report(ytst_val, y_pred))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot

y_score = opt_rf.predict_proba(xtst_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(ytst_val, y_score)

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

#x_norm_all = FScoreFeatures(xP, yP,"",0.15 ,0,False)
#print("FScore: ",x_norm_all.shape)
#xtr_val, xtst_val, ytr_val, ytst_val = train_test_split(x_norm_all, yP, test_size=0.2,random_state=42, stratify=yP)

rfecv = SVCLinearFeatures(df,xtr_val, ytr_val,"svcfeat.svc",True,3,5)
xtr_val = rfecv.transform(xtr_val)
xtst_val = rfecv.transform(xtst_val) # Validation set minimized features
print("Lin. SVC: ",xtr_val.shape)
Draw_PCA(xtr_val,ytr_val)

opt_rf1 = RFOptimize(xtr_val,ytr_val,"f1") 
#opt_rf1 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=5, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=3,
#            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
#            oob_score=False, random_state=42, verbose=0, warm_start=False)

opt_rf1.fit(xtr_val,ytr_val)
y_pred1 = opt_rf1.predict(xtst_val)
#y_pred = clf_lr.predict(X_test)
print(confusion_matrix(ytst_val, y_pred1))
print(confusion_matrix(ytst_val, y_pred))

print(classification_report(ytst_val, y_pred1))

#y_score1 = opt_rf1.predict_proba(xtst_val)[:, 1]
#
#precision, recall, thresholds = precision_recall_curve(ytst_val, y_score1)
#
##create precision recall curve
#fig, ax = plt.subplots()
#ax.plot(recall, precision, color='purple')
#
##add axis labels to plot
#ax.set_title('Precision-Recall Curve')
#ax.set_ylabel('Precision')
#ax.set_xlabel('Recall')
#
##display plot
#plt.show()
#Prec_Recall_curve(precision,recall)

y_score1 = opt_rf1.predict_proba(xtst_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(ytst_val, y_score1)

#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')

#display plot
plt.show()

from sklearn.model_selection import cross_val_score
cv_scores1 = cross_val_score(opt_rf1,xP,yP, cv=5, scoring='f1')
cv_scores = cross_val_score(opt_rf,xP,yP, cv=5, scoring='f1')

print(cv_scores)
print(cv_scores1)

Plot_avg_ROC(xP,yP,opt_rf)
Plot_avg_ROC(xP,yP,opt_rf1)