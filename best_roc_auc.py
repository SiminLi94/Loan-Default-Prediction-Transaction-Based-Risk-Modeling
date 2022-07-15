import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score,mean_squared_error, roc_curve, auc
import time
from scipy import interp
from sklearn.model_selection import StratifiedKFold
import xgboost



def mk_data(data,samples,strt_indx,stp_indx,mode):
  
    feats = 27
    
    if mode=='train':
      target = np.zeros(samples)
    else:
      target = 0
    
    features = np.zeros([samples,feats])
    
    i=0
    
    for cust in np.arange(strt_indx,stp_indx):
    
      x=0
      ts = data[2][cust][1]
      
      if mode=='train':
        target[i] = data[6][cust][1]
      else:
        target = 0
      
      cumsum_ts = np.cumsum(ts)
      max_count=np.count_nonzero(np.where(ts==np.max(ts)))
      min_count=np.count_nonzero(np.where(ts==np.min(ts)))
      
      unique_elements, counts_elements = np.unique(data[1][cust][1], return_counts=True)
      count = np.count_nonzero(counts_elements)
    
      maxv= np.where(ts==np.max(ts))
      minv= np.where(ts==np.min(ts))
      l_eq=np.where(ts>data[4][cust][1])
    
      l= data[1][cust][1][-1]
      f= data[1][cust][1][0]
      ld = data[5][cust][1]
    
      maxd = data[1][cust][1][maxv[0][-1]]
      mind = data[1][cust][1][minv[0][-1]]
    
      date_diff= datetime.strptime(l,'%Y-%m-%d') - datetime.strptime(f,'%Y-%m-%d')
      date_diff2= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(f,'%Y-%m-%d')
      date_diff3= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(l,'%Y-%m-%d')
      date_diff4= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(maxd,'%Y-%m-%d')
      date_diff5= datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(mind,'%Y-%m-%d')
    
      l_eq=np.where(ts>data[4][cust][1])
    
      if len(l_eq[0])>0:
        lda = data[1][cust][1][l_eq[0][-1]]
      else:
        lda = data[1][cust][1][0]
    
      date_diff6 =datetime.strptime(ld,'%Y-%m-%d') - datetime.strptime(lda,'%Y-%m-%d')
      loan_date=datetime.strptime(data[5][cust][1],'%Y-%m-%d')
    
      features[i,x] = np.max(ts)
      x+=1
      features[i,x] = np.min(ts)
      x+=1
      features[i,x] = count
      x+=1
      features[i,x] = max_count
      x+=1
      features[i,x] = min_count
      x+=1
      features[i,x] = np.mean(ts[np.where(ts>0)])    #1
      x+=1
      features[i,x] = np.mean(ts[np.where(ts<0)])    #2
      x+=1
      features[i,x] = np.median(ts[np.where(ts>0)])    #1
      x+=1
      features[i,x] = np.median(ts[np.where(ts<0)])    #2
      x+=1
      features[i,x] = np.std(ts[np.where(ts>0)])    
      x+=1
      features[i,x] = np.std(ts[np.where(ts<0)])
      x+=1
      features[i,x] = cumsum_ts[-1] 
      x+=1
      features[i,x] = cumsum_ts[0]   
      x+=1
      features[i,x] = np.count_nonzero(ts)
      x+=1
      features[i,x] = date_diff.days
      x+=1
      features[i,x] = date_diff2.days
      x+=1
      features[i,x] = date_diff3.days
      x+=1
      features[i,x] = date_diff4.days
      x+=1
      features[i,x] = date_diff5.days
      x+=1
      features[i,x] = date_diff6.days
      x+=1
      features[i,x] = np.count_nonzero(ts[np.where(ts>0)])
      x+=1
      features[i,x] = np.count_nonzero(ts[np.where(ts<0)])
      x+=1
      features[i,x] = np.count_nonzero(ts[np.where(ts>(data[4][cust][1]))])
      x+=1
      features[i,x] = loan_date.day #3 
      x+=1
      features[i,x] = loan_date.month #3 
      x+=1
      features[i,x] = loan_date.year #3 
      x+=1
      features[i,x] = data[4][cust][1] #3 
      
      i+=1
    
    features[np.isnan(features)] = 0
    features=np.square(features)
    
    if mode=='train':
      return features,target
    else:
      return features



data = pd.read_pickle('dataset.pkl')

train_features, train_target = mk_data(data,10000,0,10000,'train')

data = pd.read_pickle('dataset_new.pkl')

val_features, val_target = mk_data(data,500,10000,10500,'train')
test_features = mk_data(data,5000,10000,15000,'test')



clf = xgboost.XGBClassifier(n_estimators=160)

tprs = np.zeros([10,100])
aucs = np.zeros([10])
mean_fpr = np.linspace(0, 1, 100)
cv = StratifiedKFold(n_splits=10)

i=0
for train, test in cv.split(train_features, train_target):
    probas_ = clf.fit(train_features[train], train_target[train]).predict_proba(train_features[test])
    fpr, tpr, thresholds = roc_curve(train_target[test], probas_[:, 1])
    tprs[i,:] = interp(mean_fpr, fpr, tpr)
    
    roc_auc = auc(fpr, tpr)
    aucs[i] = roc_auc
    i+=1

graph =  plt.plot(mean_fpr,np.mean(tprs,axis=0))
graphx = plt.plot([0,1],[0,1])
grapht = plt.title('Mean ROC Curve : '+str(np.mean(aucs)))

plt.show()

clf.fit(train_features,train_target)
ypred=clf.predict(val_features)

print('Validation Accuracy of the mdoel: ',accuracy_score(val_target,ypred))

probas_=clf.predict_proba(val_features)
fpr, tpr, thresholds = roc_curve(val_target, probas_[:, 1])
roc_auc = auc(fpr, tpr)
graph =  plt.plot(fpr,tpr)
graphx = plt.plot([0,1],[0,1])
grapht = plt.title('Validation ROC : '+str(roc_auc))


plt.show()


probas_=clf.predict_proba(test_features)

f = open("output.txt", "w")

i=0
for cust in np.arange(10000,15000):
  o_id=str(data[0][cust][1])
  o_prob=str(probas_[i][1])
  o_str=o_id+','+o_prob+'\n'
  f.write(o_str)
  i+=1
f.close()


print('Model Score : '+str(clf.score(train_features,train_target)))

