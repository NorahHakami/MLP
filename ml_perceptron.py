#NORAHHAKAMI
import numpy as np
from sklearn.model_selection import train_test_split as sp_data
import matplotlib.pyplot as plt


X = []
y = []
with open('wdbc.data', 'r') as file:
    data = file.read().splitlines()
    for line in data:
        row = line.split(",")
        X.append([float(x) for x in row[2:]])
        y.append(1 if row[1] == 'M' else 0)
  
    X = np.array(X)
    y = np.array(y).reshape(-1,1)


c = X - np.min(X)
d = np.max(X) - np.min(X) 
X =  c/d
X_train, X_val,y_train, y_val = sp_data(X,y,test_size=0.2,random_state=32)


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

# Define the MLP class
class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.log(input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)* np.log(hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = 1 / (1 + np.exp(-self.z1))
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))
        return self.a2

    def backward(self, X, y, output):
        self.error = output - y
        self.delta2 = self.error * self.a2 * (1 - self.a2)
        self.delta1 = np.dot(self.delta2, self.W2.T) * self.a1 * (1 - self.a1)
        self.dW2 = np.dot(self.a1.T, self.delta2)
        self.db2 = np.sum(self.delta2, axis=0)
        self.dW1 = np.dot(X.T, self.delta1)
        self.db1 = np.sum(self.delta1, axis=0)

    def update(self, learning_rate):
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

    def train(self, X, y, X_val, y_val, learning_rate, epochs):
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            self.update(learning_rate)
            train_loss.append(-np.mean(y * np.log(output) + (1 - y) * np.log(1 - output)))
            val_output = self.forward(X_val)
            val_loss.append(-np.mean(y_val * np.log(val_output) + (1 - y_val) * np.log(1 - val_output)))
            train_acc.append((np.round(output) == y).mean())
            val_acc.append((np.round(val_output) == y_val).mean())
            if i % 10 == 0:
                print(f"Epoch {i + 1}/{epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")
        return train_loss, val_loss, train_acc, val_acc

# Train the MLP
mlp = MultiLayerPerceptron(input_size=30, hidden_size=6, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)

# Plot the training and validation loss and accuracy
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(train_loss, label="Train")
ax[0].plot(val_loss, label="Validation")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Cross-entropy loss")
ax[0].legend()
ax[1].plot(train_acc, label="Train")
ax[1].plot(val_acc, label="Validation")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
plt.show()


mlp = MultiLayerPerceptron(input_size=30, hidden_size=6, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=200)

# Plot the training and validation loss and accuracy
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(train_loss, label="Train")
ax[0].plot(val_loss, label="Validation")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Cross-entropy loss")
ax[0].legend()
ax[1].plot(train_acc, label="Train")
ax[1].plot(val_acc, label="Validation")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
plt.show()

mlp = MultiLayerPerceptron(input_size=30, hidden_size=6, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=1, epochs=100)

# Plot the training and validation loss 
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(train_loss, label="Train")
ax[0].plot(val_loss, label="Validation")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Cross-entropy loss")
ax[0].legend()
ax[1].plot(train_acc, label="Train")
ax[1].plot(val_acc, label="Validation")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
plt.show()

mlp = MultiLayerPerceptron(input_size=30, hidden_size=6, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.5, epochs=100)

# Plot the training and validation loss 
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(train_loss, label="Train")
ax[0].plot(val_loss, label="Validation")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Cross-entropy loss")
ax[0].legend()
ax[1].plot(train_acc, label="Train")
ax[1].plot(val_acc, label="Validation")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
plt.show()



mlp = MultiLayerPerceptron(input_size=30, hidden_size=6, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.1, epochs=100)

# Plot the training and validation loss 
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(train_loss, label="Train")
ax[0].plot(val_loss, label="Validation")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Cross-entropy loss")
ax[0].legend()
ax[1].plot(train_acc, label="Train")
ax[1].plot(val_acc, label="Validation")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
plt.show()

mlp = MultiLayerPerceptron(input_size=30, hidden_size=6, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)

# Plot the training and validation loss 
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(train_loss, label="Train")
ax[0].plot(val_loss, label="Validation")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Cross-entropy loss")
ax[0].legend()
ax[1].plot(train_acc, label="Train")
ax[1].plot(val_acc, label="Validation")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
plt.show()


accuracy = []
mlp = MultiLayerPerceptron(input_size=30, hidden_size=5, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)
accuracy.append(np.mean(val_acc))
mlp = MultiLayerPerceptron(input_size=30, hidden_size=10, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)
accuracy.append(np.mean(val_acc))
mlp = MultiLayerPerceptron(input_size=30, hidden_size=15, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)
accuracy.append(np.mean(val_acc))
mlp = MultiLayerPerceptron(input_size=30, hidden_size=20, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)
accuracy.append(np.mean(val_acc))
mlp = MultiLayerPerceptron(input_size=30, hidden_size=25, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)
accuracy.append(np.mean(val_acc))
mlp = MultiLayerPerceptron(input_size=30, hidden_size=30, output_size=1)
train_loss, val_loss, train_acc, val_acc = mlp.train(X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=100)
accuracy.append(np.mean(val_acc))

plt.plot([5,10,15,20,25,30],accuracy)
plt.xlabel('number of nodes')
plt.ylabel('accuracy')
plt.show()