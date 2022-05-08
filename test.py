import mypca
import numpy as np

x = np.array([[-1, -1, 1], [-2, -1, 1], [-3, -2, 2], [1, 1, 3], [2, 1, 1], [3, 2, 9]])
output = np.array([[2.29548456, 0.14307812],
                   [2.8334206, -0.5155151],
                   [2.97010772, -2.2054038],
                   [-1.02560371, 0.93513766],
                   [-0.03968273, 2.88758981],
                   [-7.03372643, -1.24488669]])

pca = mypca.SimplePCA()
transformation = pca.fit_transform(x)


check_parameters = np.round(transformation, 2) == np.round(output, 2)
if np.all(check_parameters == True):
    print("mypca is working!!")
else:
    print("The result is different from expected, please check the again the implementation")
