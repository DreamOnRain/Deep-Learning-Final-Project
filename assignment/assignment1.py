import numpy as np
import matplotlib.pyplot as plt

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
# path = '../data'
# wget.download(url, path)

datafile = ('../data/housing.data')
data = np.fromfile(datafile, sep=' ')
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])
print(data.shape)

#rescale
def scale_data(X):
    X_scale = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
    return X_scale

#training data ratio
ratio = 0.8
offset = int(data.shape[0] * ratio)

x = data[:,:-1]
y = data[:,-1:]
x = scale_data(x)
X_train = x[:offset]
X_test = x[offset:]
Y_train = y[:offset]
Y_test = y[offset:]

print(f'The shape of the training data is {X_train.shape[0]} * {X_train.shape[1]}')
print(f'The shape of the test set is {X_test.shape[0]} * {X_test.shape[1]}')

# housedatadf = pd.DataFrame(data=X_train, columns=feature_names[:-1])
# housedatadf['target']=Y_train
# datacor = np.corrcoef(housedatadf.values, rowvar=0)
# datacor=pd.DataFrame(data=datacor, columns=housedatadf.columns, index=housedatadf.columns)
# plt.figure(figsize=(15, 10))
# ax = sns.heatmap(datacor, square=True, annot=True, fmt='.3f', linewidths=5, cmap='YlGnBu', cbar_kws={'fraction':0.046, 'pad':0.03})
# plt.show()


#build one linear layer module
class MLP(object):
    def __init__(self, num_of_weights, learning_rate=0.01):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.weight = np.random.normal(0, 0.01, (num_of_weights, 1))
        self.bias = 0

    def mse_loss(self, y_pred, y):
        loss_ = np.mean(np.square(y_pred - y))
        return loss_

    def forward(self, x):
        out_ = np.dot(x, self.weight) + self.bias
        return out_

    def backward(self, x, y_pred, y):
        gradient_weight = np.mean((y_pred - y) * x, axis=0)
        gradient_weight = gradient_weight[:, np.newaxis]
        gradient_bias = np.mean(y_pred - y, axis=0)

        self.weight -= self.learning_rate * gradient_weight
        self.bias -= self.learning_rate * gradient_bias

    def train(self, X, Y, num_epochs, batch_size):
        n_samples = len(X)
        losses = []
        for epoch_id in range(num_epochs):
            shuffle = np.random.permutation(n_samples)
            X_batches = np.array_split(X[shuffle], n_samples / batch_size)
            Y_batches = np.array_split(Y[shuffle], n_samples / batch_size)
            iter_id = 0
            for batch_x, batch_y in zip(X_batches, Y_batches):
                y_prediction = self.forward(batch_x)
                loss = self.mse_loss(y_prediction, batch_y)
                self.backward(batch_x, y_prediction, batch_y)
                losses.append(loss)
                iter_id += 1
        return losses

#train
network = MLP(13, 0.1)
losses = network.train(X_train, Y_train, num_epochs=50, batch_size=100)
plot_x = np.arange(len(losses)) #np.arange: Return evenly spaced values within a given interval.
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

# test
Y_pred = network.forward(X_test)
Y_pred = np.squeeze(Y_pred) #np.squeeze: Remove axes of length one from a.
Y_test = np.squeeze(Y_test)
index = np.argsort(Y_test) #np.argsort: Returns the indices that would sort an array.
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(Y_test)), Y_test[index], 'r', label='original=y')
plt.scatter(np.arange(len(Y_test)), Y_pred[index], s=3, c='b', label='prediction')
plt.legend(loc='upper left') #legend: drawing legends associated with axes and/or figures
plt.grid()
plt.xlabel('index')
plt.ylabel('y')
plt.show()