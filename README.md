# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1..Import the packages required.

2.Read the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary and predict the Regression value .


## Program:

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Gayathri A
RegisterNumber:  212221230028

```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad
```

## Output:

![ml51](https://user-images.githubusercontent.com/94154854/204433496-2ad30ef7-5d17-4f04-8145-ef23ad6dd876.png)

![ml52](https://user-images.githubusercontent.com/94154854/204433522-80070626-d4a8-448a-bac8-2c9c6d9d1a82.png)

![ml53](https://user-images.githubusercontent.com/94154854/204433538-e76c7ffb-6474-4b02-8963-946b205e8bcc.png)

![ml54](https://user-images.githubusercontent.com/94154854/204433557-ed112f7d-c2f3-4ad9-90e3-2c3243210314.png)

![ml55](https://user-images.githubusercontent.com/94154854/204433603-608ca88a-bbdb-4035-9cc3-49e2b44c2cb7.png)

![ml56](https://user-images.githubusercontent.com/94154854/204433618-8510068b-b331-44f1-915d-990303847cd9.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

