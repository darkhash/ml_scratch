import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=10000):
        self.m = np.random.randn()
        self.c = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        pass

    def train(self, training_data):
        self.training_data = training_data

        for i in range(self.epochs):
            m_grad, c_grad = self.gradient()
            self.m += self.learning_rate * m_grad
            self.c += self.learning_rate * c_grad

            pass
        print(self.m, self.c)

    def predict(self, x):
        return np.dot(self.m, x) + self.c

    def gradient(self):
        x, y = self.training_data
        prediction = self.predict(x)
        c_grad = np.sum((y - prediction)) / len(y)
        m_grad = x.T.dot(y - prediction) / len(y)
        return m_grad, c_grad

    def loss(self, x, y):
        return (1 / 2) * (y - (self.m * x + self.c)) ** 2



if __name__ == '__main__':
    x = np.linspace(0, 1, 20)
    y = 2 * x + 5
    # add some randomness
    y = y + np.random.randint(0, 10)

    training_data = np.array(x), np.array(y)
    plt.scatter(x, y)

