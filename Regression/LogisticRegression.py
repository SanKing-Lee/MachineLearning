from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def newton_method(x, y):
    m, n = np.shape(x)
    # the likely x
    xllh = np.c_[x, np.ones(m)]
    print("xllh:")
    print(xllh)
    # number of iterations
    # iterations = 2000
    # beta = np.array([0, 0, 1])
    # for i in range(iterations):
    #     beta1sum = [0, 0, 0]
    #     beta2sum = 0
    #     for j in range(m):
    #         numerator = np.exp(np.dot(np.transpose(beta), xllh[j]))
    #         p1 = numerator/(1+numerator)
    #         beta1sum -= np.dot(xllh[j], y[j]-p1)
    #         beta2sum += np.dot(xllh[j], np.transpose(xllh[j]))*(1-p1)
    #     beta = beta - np.dot((1/beta2sum), beta1sum)
    iterations = 0
    beta = np.array([0, 0, 4])
    beta_old = np.array([0, 0, 4])
    # if the change is bigger than 0.00001 or the first iterate, then keep on iterating
    while (np.abs((beta-beta_old)[1]) > 0.00001) or (iterations == 0):
        beta_old = beta
        beta1sum = [0, 0, 0]
        beta2sum = 0
        for j in range(m):
            numerator = np.exp(np.dot(np.transpose(beta), xllh[j]))
            p1 = numerator/(1+numerator)
            beta1sum -= np.dot(xllh[j], y[j]-p1)
            beta2sum += np.dot(xllh[j], np.transpose(xllh[j]))*(1-p1)
        beta = beta_old - np.dot((1/beta2sum), beta1sum)
        iterations += 1
    print("beta: ")
    print(beta)
    print("iterations: ")
    print(iterations)
    return beta


def prediction(beta, x_test):
    # bigger than 0.4 then good watermelon, otherwise bad
    threshold = 0.4
    m, n = np.shape(x_test)
    x_test = np.c_[x_test, np.ones(m)]
    # the list to store y_predict
    predict = []
    for i in range(m):
        t = np.dot(np.transpose(beta), x_test[i])
        y = 1/(1+(np.exp(-t)))
        if y > threshold:
            y = 1
        else:
            y = 0
        predict.append(y)
    return np.array(predict)


def main():
    # load data
    data = np.loadtxt("watermelon_dataset.txt", delimiter="\t")
    x = data[:, 1:3]
    y = data[:, 3]
    # split train set and test set
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.4, random_state=0)
    print(x)
    print(y)
    # get the parameter
    beta = newton_method(x_train, y_train)
    y_pred = prediction(beta, x_test)
    print("y_test:")
    print(y_test)
    print("y_pred:")
    print(y_pred)
    # confusion matrix
    print("confusion matrix: ")
    print(metrics.confusion_matrix(y_test, y_pred))
    # classifier
    print("classifier: ")
    print(metrics.classification_report(y_test, y_pred))

    # draw the plot
    plt.figure(1)

    # first subplot to draw the scatters of all the watermelon
    f1 = plt.subplot(211)
    # label of the axis
    plt.xlabel('Density')
    plt.ylabel('Sugar')
    # title
    plt.title('Watermelon')
    plt.xlim(0.2, 0.8)
    plt.ylim(0, 0.5)
    # scatter point
    plt.scatter(x[y == 0, 0], x[y == 0, 1], s=50, marker='o', color='k',
                label='bad')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], s=50, marker='o', color='g',
                label='good')
    plt.legend(loc='upper left')

    # second subplot to draw the prediction watermelon
    plt.subplot(212)
    plt.xlabel('Density')
    plt.ylabel('Sugar')
    # title
    plt.title('Prediction')
    plt.xlim(0.2, 0.8)
    plt.ylim(0, 0.5)
    plt.scatter(x_test[y_pred == 0, 0], x_test[y_pred == 0, 1], s=50, marker='o', color='r', label='predict bad')
    plt.scatter(x_test[y_pred == 1, 0], x_test[y_pred == 1, 1], s=50, marker='o', color='b', label='predict good')

    # the location of the legend
    plt.legend(loc='upper left')
    plt.show()


main()
