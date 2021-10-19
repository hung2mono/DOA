#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import sklearn
import sklearn.model_selection
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score, log_loss
import seaborn as sns # data visualization library

data = pd.read_csv('../Data/data.csv')
X = data.iloc[:,:90]
y = data.iloc[:,90:]
N_FEATURES = 90

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, test_size=0.3)

print(np.shape(x_train))
print(np.shape(y_train))

def xNumpyToTensor(array):
    array = np.array(array, dtype=np.float32)
    return Variable(torch.from_numpy(array)).type(torch.FloatTensor)

def yNumpyToTensor(array):
    array = np.array(array.astype(int))
    return Variable(torch.from_numpy(array)).type(torch.FloatTensor)

x_tensor_train = xNumpyToTensor(x_train)
y_tensor_train = yNumpyToTensor(y_train)
x_tensor_test = xNumpyToTensor(x_test)
y_tensor_test = yNumpyToTensor(y_test)

# Neural Network parameters
DROPOUT_PROB = 0.9
input_size_ss = 90
output_size = 121
LR = 0.001
MOMENTUM= 0.99
dropout = torch.nn.Dropout(p=1 - (DROPOUT_PROB))
hidden_size_ss = [int(2 / 3 * input_size_ss), int(4 / 9 * input_size_ss), int(1 / 3 * input_size_ss)]
hiddenLayer1Size=int(2 / 3 * input_size_ss)
hiddenLayer2Size=int(4 / 9 * input_size_ss)
hiddenLayer3Size=int(1 / 3 * input_size_ss)


#Neural Network layers
linear1=torch.nn.Linear(N_FEATURES, hiddenLayer1Size, bias=True)
linear2=torch.nn.Linear(hiddenLayer1Size, hiddenLayer2Size)
linear3=torch.nn.Linear(hiddenLayer2Size, hiddenLayer3Size)
linear4=torch.nn.Linear(hiddenLayer3Size, output_size)


softmax = torch.nn.Softmax()
threshold = nn.Threshold(0.5, 0)
tanh=torch.nn.Tanh()
relu=torch.nn.LeakyReLU()

#Neural network architecture
net = torch.nn.Sequential(linear1,nn.BatchNorm1d(hiddenLayer1Size),relu,
                          linear2,dropout,relu,
                          linear3,dropout,relu,
                          linear4,dropout,relu,
                          softmax
                          )
optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=5e-3)
loss_func=torch.nn.BCELoss()
epochs = 200
all_losses = []

#Training in batches
for step in range(epochs):
    out = net(x_tensor_train)  # input x and predict based on x
    cost = loss_func(out, y_tensor_train)
    optimizer.zero_grad()  # clear gradients for next train
    cost.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if step % 5 == 0:
        loss = cost.data
        all_losses.append(loss)
        print(step, cost.data.cpu().numpy())
        # RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays).
        # Use .cpu() to move the tensor to host memory first.
        prediction = (net(x_tensor_test).data).float()  # probabilities
        #         prediction = (net(X_tensor).data > 0.5).float() # zero or one
        #         print ("Pred:" + str (prediction)) # Pred:Variable containing: 0 or 1
        #         pred_y = prediction.data.numpy().squeeze()
        pred_y = prediction.cpu().numpy().squeeze()
        target_y = y_tensor_test.cpu().data.numpy()
        print('LOG_LOSS={} '.format(log_loss(target_y, pred_y)))

    # Evaluating the performance of the model
# matplotlibinline
plt.plot(all_losses)
plt.show()
pred_y = pred_y > 0.5
print('f1 score', f1_score(target_y, pred_y))
print('accuracy', accuracy_score(target_y, pred_y))
cm = confusion_matrix(target_y, pred_y)
sns.heatmap(cm, annot=True)