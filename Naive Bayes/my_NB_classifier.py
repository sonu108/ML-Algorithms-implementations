
# coding: utf-8

# In[115]:

import pandas as pd
from sklearn import cross_validation
import math
import warnings
warnings.filterwarnings('ignore')


# In[116]:

full_data = pd.read_csv("diabetes.csv")


# In[117]:

X = full_data.drop('Outcome',1)
y = full_data['Outcome']


# In[214]:

y.head()


# In[181]:

def calculateProbability(x, mean, stdev):
    #print x , type(x)
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# In[182]:

def class_summarize(full_data):
    class_dict = {}
    for cls in full_data['Outcome'].unique():
        class_dict[cls] = full_data[full_data['Outcome']==cls]
        class_dict[cls].drop(['Outcome'] , 1 , inplace = True)
    for key in class_dict:
        temp = class_dict[key]
        summ = []
        for i in temp:
            summ.append((temp[i].mean() , temp[i].std()))
        class_dict[key] = summ
    return class_dict
    
            

    


# In[183]:

class_dict = class_summarize(full_data)
print len(full_data[full_data['Outcome']==1])


# In[184]:

def calculate_class_prob(class_dict , input_set):
    probs = {}
    for cls , cls_info in class_dict.iteritems():
        probs[cls] = 1
        #print len(cls_info)
        for i in range(len(cls_info)):
            mean , std = cls_info[i]
            x = input_set[i]
            probs[cls] *= calculateProbability(x , mean , std)
    return probs


# In[185]:

def predict(class_dict , input_set):
    probabilities = calculate_class_prob(class_dict , input_set)
    result_label , result_prob = None , -1
    for cls , prob in probabilities.iteritems():
        if result_label is None or prob > result_prob:
            result_label = cls
            result_prob = prob
    return result_label
        


# In[186]:

def get_all_prediction(class_dict , testset):
    results = []
    for i in testset:
        r = predict(class_dict , i)
        results.append(r)
    return results


# In[196]:

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0


# In[238]:

X_train , X_test , y_train , y_test = cross_validation.train_test_split(X,y,test_size = 0.37)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train , y_train)
print X_train.shape , X_test.shape , y_train.shape , y_test.shape
y_pred = gnb.predict(X_test)
c = 0
y_temp = y_test.values
for i in range(len(y_temp)):
    if y_temp[i]==y_pred[i]:
        c+=1

print c*1.0/len(y_temp)*100


# In[239]:

X_train['Outcome'] = y_train
class_dict = class_summarize(X_train)
#pd.to_numeric(X_test)
X_test = X_test.astype(float)
#print X_test
#print X_test.head()
#print X_test.loc[70]
#X_test.head()


# In[240]:

#print class_dict
prediction = get_all_prediction(class_dict,X_test.values)


# In[241]:

getAccuracy( y_test.values , prediction)


# In[ ]:




# In[ ]:




# In[ ]:



