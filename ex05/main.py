import numpy as np
from mylinearregression import MyLinearRegression as MyLR


X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR(np.array([[1.], [1.], [1.], [1.], [1]]))
y_hat = mylr.predict_(X)
print(y_hat)
print(mylr.loss_elem_(Y, y_hat))
print(mylr.loss_(Y, y_hat))


mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.thetas)
y_hat = mylr.predict_(X)
print(y_hat)
print(mylr.loss_elem_(Y, y_hat))
print(mylr.loss_(Y, y_hat))