## Multilayer Neural Network Binary Classifier

This library is written for educational purposes only.
The architecture of neural network is inspired by Andrew Ng course on deep learning.

**class MultiLayerNN**(  
    dimensions=[1],  
    cost_func_name='logloss',  
    activation_functions=['sigmoid'],  
    learning_rate=0.001,  
    n_iter=20000,  
    verbose=True   
)  
This multi layer network implementation supports:  
-any number of layers  
-any layer size  
-any activation functions ('relu', 'sigmoid', 'tanh', 'leaky_relu') for each layer  

It initilizes weights using He initialization (He et al. 2015)  

It doesn't support regularization yet, but  will support in nearest future.  

Example of use for solving binary classification problem on iris dataset.  

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

print(nn.score(X_test, y_test))
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
0.9555555555555556
```