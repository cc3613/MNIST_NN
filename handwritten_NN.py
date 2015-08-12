import cPickle, gzip
import numpy as np
import time

start_time=time.time()
#from deeplearning.net
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#combine the sets for reshuffling, need this for k-fold cross validation
combine_set=(np.r_['0,2', train_set[0], valid_set[0]], np.r_['-1', train_set[1], valid_set[1]])

def reshuffle(data_set):
	indices=np.random.permutation(data_set[0].shape[0])
	train_idx, valid_idx = indices[:50000], indices[50000:]
	train_set, valid_set = (data_set[0][train_idx, :], data_set[1][train_idx]), (data_set[0][valid_idx, :], data_set[1][valid_idx])

	return train_set, valid_set

#using hyperbole tan instead of sigmoid function


def nonlin(x, deriv=False):
	if (deriv ==True):
		#return 1-np.tanh(x)*np.tanh(x)
		return 1-x*x
	return np.tanh(x)



#seed random numbers to make calculation
#deterministic (just a good practice)
np.random.seed(1)

#set pair aking for train or test
def pair_making(data_set):
	#splitting the two arrays from data_set
	data_set_x, data_set_y = data_set
	#converting the numerics into booleans, location of '1' indicates what
	#the number is. 10 digits total.
	data_set_y = np.eye(10)[:, data_set_y].T
	return data_set_x, data_set_y



#forward propogation (with added bias neurons)
def forward_prop(set_x, m):
	#append ones
	l0=np.c_[np.ones((m,1)), set_x]
	#dot the train set and the first set of weight to forward proporgate
	#to first layer (hidden layer)
	l1=nonlin(np.dot(l0, syn0))
	#append 1's again
	l1=np.c_[np.ones((m,1)), l1]
	#forward to second layer (output layer)
	l2=nonlin(np.dot(l1, syn1))

	return l0, l1, l2, l2.argmax(axis=1)


#cross validation for n times
n = 10
num_iter=100
cross_result=[]
for k in range(n):
	#select train_set
	train_set, valid_set=reshuffle(combine_set)
	
	#testing if the algo runs outside of the def
	train_x, train_y = pair_making(train_set)
	syn0 = (2*np.random.random((785,100)) - 1)/10
	syn1 = (2*np.random.random((101,10)) - 1)/10
	alpha = 0.05
	beta = 0.001
	momentum = 0.99
	m = len(train_x)
	velocities0 = np.zeros(syn0.shape)
	velocities1 = np.zeros(syn1.shape)
	for i in range (num_iter):
		#forward propogation
		l0, l1, l2, _ = forward_prop(train_x, m)

		#calculate error
		l2_error=l2 - train_y
		if i%10==0:
			print("Error " + str(i) + ": " + str(np.mean(np.abs(l2_error))))

		#apply the sigmoid (or deriv tanh) to the error
		l2_delta=l2_error * nonlin(l2, deriv=True)

		#backward propagation
		l1_error = l2_delta.dot(syn1.T)
		l1_delta = l1_error * nonlin(l1, deriv=True)
		l1_delta = l1_delta[:, 1:]

		#update the weights
		v0= velocities0
		v1= velocities1

		velocities0 = velocities0* momentum - alpha * (l0.T.dot(l1_delta)/m)
		velocities1 = velocities1* momentum - alpha * (l1.T.dot(l2_delta)/m)

		syn1 += -v1 * momentum +(1+momentum) * velocities1 - alpha * beta * syn1 /m
		syn0 += -v0 * momentum +(1+momentum) * velocities0 - alpha * beta * syn0 /m
	
	#test on valid_set
	valid_x, valid_y = pair_making(valid_set)
	predictions=[]
	corrects=[]
	for j in range(len(valid_x)):
		_,_,_,rez=forward_prop([valid_x[j,:]],1)
		predictions.append(rez[0])
		corrects.append(valid_y[j].argmax())

	predictions=np.array(predictions)
	corrects=np.array(corrects)

	cross_result.append(np.sum(predictions == corrects).astype('float64') / len(valid_x))


cross_avg=(sum(cross_result)/n)

print ("program took %s seconds" % (time.time() - start_time))

#To test on the test set
#test_x, test_y = pair_making(test_set)
#predictions = []
#corrects = []
#for i in range(len(test_x)):
#	_,_,_, rez = forward_prop([test_x[i,:]],1)

#	predictions.append(rez[0])
#	corrects.append(test_y[i].argmax())

#predictions = np.array(predictions)
#corrects = np.array(corrects)

#print "Accuracy is " + str(np.sum(predictions == corrects).astype('float64') / len(test_x))



	



