# Multilayer Neural Network Binary Classifier

This library is written for educational purposes only.
The architecture of neural network is inspired by Andrew Ng course on deep learning.
This multi layer network implementation supports:  
  * any number of layers  
  * any layer size  
  * any activation functions ('relu', 'sigmoid', 'tanh', 'leaky_relu') for each layer  

It initilizes weights using He initialization (He et al. 2015)  

It doesn't support regularization yet, but  will support in nearest future. 
## class MultiLayerNN 
(  
&nbsp;&nbsp;&nbsp;&nbsp;dimensions=[1],  
&nbsp;&nbsp;&nbsp;&nbsp;cost_func_name='logloss',  
&nbsp;&nbsp;&nbsp;&nbsp;activation_functions=['sigmoid'],  
&nbsp;&nbsp;&nbsp;&nbsp;learning_rate=0.001,  
&nbsp;&nbsp;&nbsp;&nbsp;n_iter=20000,  
&nbsp;&nbsp;&nbsp;&nbsp;verbose=True   
)  
 

### Parameters:
  * **dimentions**: list/tuple. For example [2, 3, 1] means, that first
  layer consists of 2 neurons, while the second consists of 3 neurons and output layer is 1 neuron.
  * **cost_func_name**: string, cost function used (currently only 'logloss' is supported)
  * **activation_functions**: list/tuple/string, if list or tuple, then represents separate function for each layer, otherwise string 
  if all layers use the same activation function
  * **learning_rate**: float, learning rate for gradient descent
  * **n_iter**: int, max number of iterations for gradient descent
  * **verbose**: bool

### Attributes:
  * **params**: dict, W and b matrices of layers
  * **cache**: dict, a's (activations) and z's (linear component) of layers
  * **grads**: dict, gradients of layers
  * **dept**: int, depth of NN
  * **cost**: float, cost 

### Methods:
  * **fit(self, X, y)**: build and train NN
  * **predict(self, X)**: return prediction for sample X
  * **predict_proba**(self, X): predict class probabilities on the given data and lables
  * **score(self, X, Y)**: return the mean accuracy of the given data and labels

### Example of use for solving binary problem on iris dataset.  

```python
from NumpyNN import MultiLayerNN
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


X, y = load_iris(return_X_y=True)
y = np.where(y == 1, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.6,
    random_state=42
)

nn_config = {
    'dimensions': [3, 3, 1],
    'activation_functions': ['relu', 'leaky_relu', 'sigmoid'],
    'cost_func_name': 'logloss',
    'learning_rate': 0.01,
    'n_iter': 20000
}
nn = MultiLayerNN(
    **nn_config
)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)

print('Test accuracy: {}'.format(nn.score(X_test, y_test)))
```
```bash
iter :1000, cost: 38.94155952212185
iter :2000, cost: 38.88808992051157
iter :3000, cost: 38.85700717435183
iter :4000, cost: 38.83165890867717
iter :5000, cost: 38.800271577808765
iter :6000, cost: 38.74445041555928
iter :7000, cost: 38.61061687331272
iter :8000, cost: 38.16731825030644
iter :9000, cost: 36.99581299808568
iter :10000, cost: 32.619242942269274
iter :11000, cost: 29.122315683663253
iter :12000, cost: 21.541892975526636
iter :13000, cost: 14.777152787646038
iter :14000, cost: 10.89770674107378
iter :15000, cost: 9.043484158634632
iter :16000, cost: 7.98640809743567
iter :17000, cost: 7.3062872650019175
iter :18000, cost: 6.799039660432962
iter :19000, cost: 6.386827466017606
iter :20000, cost: 6.042268810876664
Test accuracy: 0.9555555555555556
```