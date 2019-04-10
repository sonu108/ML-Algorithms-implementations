from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import random

style.use('ggplot')

#xs = np.array([1,2,3,4,5,6],dtype=np.float64)
#ys = np.array([5,4,6,5,6,7],dtype=np.float64)

#m =  
#b = mean(ys) - m.mean(xs)

def create_dataset(hm, variance , step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val+ random.randrange(-variance,variance)
		ys.append(y)
		if correlation and correlation=='pos':
			val+=step
		if correlation and correlation=='neg':
			val-=step
	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)





def best_fit_slop_and_intercept(xs,ys):
	m = ( (xs.mean() * ys.mean()) - (xs*ys).mean() )/ ((xs.mean()*xs.mean()) - (xs*xs).mean() )
	b = ys.mean() - m*(mean(xs))
	return m,b

def squared_error(ys_orig,ys_line):
	return sum((ys_orig-ys_line)**2)

def coefficient_of_determination(ys_orig,ys_line):
	y_mean_line = [mean(ys_orig) for i in ys_orig]
	squared_error_reg = squared_error(ys_orig,ys_line)
	squared_error_y_mean = squared_error(ys_orig,y_mean_line)
	return 1 - (squared_error_reg / squared_error_y_mean)


xs,ys = create_dataset(40,10,2,correlation='neg')



m , b= best_fit_slop_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]

r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared,metrics.r2_score(ys,regression_line))
'''
xs = np.array([1,2,3,4,5,6])
ys = np.array([5,4,6,5,6,7])
print(xs,xs.shape)
xs = xs.reshape(len(xs),-1)
ys = ys.reshape(len(ys),-1)
print ys,ys.shape
clf = LinearRegression()
clf.fit(xs,ys)'''

#model_result = clf.predict(xs)
#print(model_result)
predict_x = 8
predict_y = (m*predict_x)+b
#print predict_y

#print(clf.coef_,m,b)
plt.scatter(xs,ys)
plt.plot(xs,regression_line,color='b')
plt.scatter(predict_x, predict_y , s=200 , color='g')
plt.show()
#plt.plot(xs,model_result,color='r')
#plt.show()