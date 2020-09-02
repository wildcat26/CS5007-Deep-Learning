test = open("Test.csv", "a")
train = open("Train.csv", "a")
testList = []
trainList = []
path = "dataset"
for i in range (1, 8):
    f1 = open(path + str(i) + "/Test" + str(i) + ".csv", "r")
    f2 = open(path + str(i) + "/Train" + str(i) + ".csv", "r")
    testList = f1.readlines()
    trainList = f2.readlines()
    test.writelines(testList)
    train.writelines(trainList)
    f1.close()
    f2.close()
test.close()
train.close()
