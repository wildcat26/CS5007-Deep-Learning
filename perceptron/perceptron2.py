import matplotlib.pyplot as plt
import sys
import numpy as np

from sklearn.preprocessing import MinMaxScaler

FEATURES = 4
TRAIN_SIZE = 1097
TEST_SIZE = 1372 - 1097

def add(a, b, mul):
    i = 0
    for val in b:
        a[i] += (val * mul)
        i = i + 1
    return a

def perceptron(data, Y, alpha, N):
    global FEATURES, TRAIN_SIZE
    wT = np.random.random(FEATURES + 1)
    error, okay = [], []
    correct, itr = 0, 0
    while (correct < TRAIN_SIZE and itr < N):
        correct = 0
        itr += 1
        idx = 0
        for x in data[:TRAIN_SIZE]:
            dotPro = np.dot(wT, x)
            if (dotPro > 0 and Y[idx] < 0):
                wT = add(wT, x, -1*alpha)
            elif (dotPro < 0 and Y[idx] > 0):
                wT = add(wT, x, alpha)
            else:
                correct += 1
            idx += 1
        error.append(TRAIN_SIZE - correct)
        okay.append(correct)
    print("Iterations needed = ", itr)
    return wT, error, okay, itr


def readFile():
        n = 0
        trn = open("Dataset_Question2.csv", "r")
        lines = trn.readlines()
        data, Y = [], []
        for words in lines:
            n = n + 1
            temp = words.split(",")
            print("temp ", temp)
            xx = []
            for val in temp:
                xx.append(float(val))
            xx[-1] = 1 # bias term!
            y = (int(2 * float(temp[-1]) - 1))
            Y.append(y)
            print("y = ", y)
            data.append(xx)
            print(xx)
        trn.close()
        min_max_scaler = MinMaxScaler()
        data = min_max_scaler.fit_transform(data)
        return data, Y, n


#def drawLine(lx, rx, wT, xx1, yy1, xx2, yy2, val):
#        x = np.linspace(lx, rx, 100)
#        y = []
#        # w0x0 + w1x1 + w2*1 = 0 => x1 = -(w0x0 + w2)/w1
#        if wT[1] == 0:
#           y = np.linspace(lx, rx, 100)
#        else:
#            m = -1 * wT[0]/wT[1]
#            c = -1 * wT[2]/wT[1]
#            for i in range(100):
#                y.append(x[i] * m + c)
#        plt.scatter(xx1, yy1, color="red")
#        plt.scatter(xx2, yy2, color="blue")
#        plt.plot(x, y)
#        if val == 0:
#            plt.title("Training")
#        else:
#            plt.title("Testing")
#        plt.show()


def plotError(wT, error, okay, data, Y, itr):
        global FEATURES
        xx = np.linspace(1, itr, itr)
        plt.plot(xx, error)
        plt.title("Classification error vs Iteration [Training]")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Classification Error")
        plt.show()

        plt.plot(xx, okay)
        plt.title("Classification accuracy vs Iteration [Training]")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Classification Accuracy")
        plt.show()


def main():
        data, Y, n = readFile()
        wT, error, okay, itr = perceptron(data, Y, 0.001, 1000)
        # TRAINED NOW !
        print(wT)
        plotError(wT, error, okay, data, Y, itr)

        idx = TRAIN_SIZE
        truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0
        for x in data[TRAIN_SIZE:]:
            dotPro = 0.0
            # print(type(wT), type(x))
            for p in range(FEATURES + 1):
                dotPro += (wT[p] * x[p])
            if ((dotPro > 0) and (Y[idx] > 0)):    # true +ve
                truePos += 1
            elif ((dotPro < 0) and (Y[idx] < 0)): # true -ve
                trueNeg += 1
            elif ((dotPro < 0) and (Y[idx] > 0)): # is pos, but classified -ve
                falseNeg += 1
            else:
                falsePos += 1
            idx += 1
        classAcc = (truePos + trueNeg)
        print("Classification accuracy = ", classAcc, " out of TEST_SIZE = ", TEST_SIZE)
        print("True +ve : ", truePos)
        print("True -ve : ", trueNeg)
        print("False +ve : ", falsePos)
        print("False -ve : ", falseNeg)

if __name__ == "__main__":
    main()
