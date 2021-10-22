import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import pandas as pd
import sklearn.model_selection
data = pd.read_csv('../Data/data.csv')
X = data.iloc[:, :90]
y = data.iloc[:, 90:]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, test_size=0.3)
N_FEATURES = 90
out_put = 121
number_hidden = 256

def xNumpyToTensor(array):
    array = np.array(array, dtype=np.float32)
    return Variable(torch.from_numpy(array)).type(torch.FloatTensor)


def yNumpyToTensor(array):
    array = np.array(array.astype(int))
    return Variable(torch.from_numpy(array)).type(torch.FloatTensor)


x_train = xNumpyToTensor(x_train)
y_train = yNumpyToTensor(y_train)
x_test = xNumpyToTensor(x_test)
y_test = yNumpyToTensor(y_test)

class LSTMPrdiction(nn.Module):
    def __init__(self, n_hidden=number_hidden):
        super(LSTMPrdiction, self).__init__()
        self.n_hidden = n_hidden
        # lstm1, lstm2, linear
        self.lstm1 = nn.LSTMCell(N_FEATURES, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, out_put)

    def forward(self, x, future=0):
        outputs = []
        n_samples = 138600

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        for input_t in x.split(90, dim = 1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t, c_t))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t, c_t))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs

if __name__ =="__main__":
    model = LSTMPrdiction()
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=5e-3)

    n_epoch = 10
    for i in range(n_epoch):
        print("epoc: ",(i+1))

        def closure():
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(out, y_train)
            print("loss: ",loss)
            loss.backward()
            return loss
        optimizer.step(closure)

        # with torch.no_grad():
        #     future = 1
        #     pred = model(x_test, future = future)
        #     loss = criterion(pred[:,:-future], y_test)
        #     print("test_loss", loss.item())
