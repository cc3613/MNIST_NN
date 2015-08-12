#running NN on MNIST, no k-fold cross validation

import cPickle, gzip
import numpy as np
import time

start_time=time.time()
#from deeplearning.net
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

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


#getting samples and tags from the set
train_x, train_y = pair_making(train_set)
#using 100 neurons for the hidden layer
#adding one more dimension for each weight as later a column of 1's will
#be appended
syn0 = (2*np.random.random((785,100)) - 1)/10
syn1 = (2*np.random.random((101,10)) - 1)/10

#setting learning rate (alpha), weight decay (beta), and momentum (Nesterov)
alpha = 0.05
beta = 0.001
momentum = 0.99

m = len(train_x) # number of training samples

#initiate velocities for later calculation
velocities0 = np.zeros(syn0.shape)
velocities1 = np.zeros(syn1.shape)
num_iter=1000
for i in range (num_iter):
	#forward propogation
	l0, l1, l2, _ = forward_prop(train_x, m)

	#calculate error
	l2_error=l2 - train_y
	if i%100==0:
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

#To test on the test set
test_x, test_y = pair_making(test_set)
predictions = []
corrects = []
for i in range(len(test_x)):
	_,_,_, rez = forward_prop([test_x[i,:]],1)

	predictions.append(rez[0])
	corrects.append(test_y[i].argmax())

predictions = np.array(predictions)
corrects = np.array(corrects)

print "Accuracy is " + str(np.sum(predictions == corrects).astype('float64') / len(test_x))
print ("Program took %s seconds" % (time.time() - start_time))
