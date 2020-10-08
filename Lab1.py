
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt


random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
ALGORITHM = "tf_net"
#ALGORITHM = "tf_conv"

#DATASET = "mnist_d"
#DATASET = "mnist_f"
#DATASET = "cifar_10"
DATASET = "cifar_100_f"
#DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 20
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 6):
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(784, activation=tf.nn.sigmoid), tf.keras.layers.Dense(512, activation=tf.nn.sigmoid),
         tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    yTrainmod = np.zeros(y.shape[0])
    for i in range(0, yTrainmod.size):
        yTrainmod[i] = y[i].argmax(axis=0)
    model.fit(x, yTrainmod, epochs=5)
    return model
# If statements for different datasets

def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.25):
    model = keras.Sequential()
    inShape = (IH, IW, IZ)
    lossType = keras.losses.categorical_crossentropy
    if DATASET == "cifar_100_f" or DATASET == "cifar_100_c":
        # 36.20 for cifar_100_f
        # 50.88 for cifar_100_c
        eps = 17
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=inShape, padding='same'))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(dropRate))
        model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(dropRate))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    elif DATASET == "mnist_d":
        # 99.05
        eps = 6
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation= "relu", input_shape=inShape))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(keras.layers.Dropout(dropRate))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    elif DATASET == "mnist_f":
        # 92.08
        eps = 11
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=inShape, padding='same'))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(dropRate))
        model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(dropRate))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    elif DATASET == "cifar_10":
        # 72.16
        eps = 11
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=inShape, padding='same'))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(dropRate))
        model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(dropRate))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation="relu"))
        model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

    model.compile(optimizer='adam', loss=lossType)
    model.fit(x, y, epochs=eps)
    return model

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        mnist = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_100_f":
        mnist = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data(label_mode="fine")
    elif DATASET == "cifar_100_c":
        mnist = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data(label_mode="coarse")
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")

def modPreds(preds):
    modout = np.zeros(preds.shape)
    for i in range(0, preds.shape[0]):
        modout[i][preds[i].argmax(axis=0)] = 1
    return modout

def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    # datasets = ["mnist_d", "mnist_f", "cifar_10", "cifar_100_c", "cifar_100_f"]
    # annResults = [97.74, 74.25, 39.50, ]
    # cnnResults = [99.05, 93.12, 76.69, 55.28, 41.12]
    # ypos = np.arange(len(datasets))
    # plt.xticks(ypos, datasets)
    # plt.bar(ypos, annResults)
    # plt.savefig("ANNAccuracy.pdf")
    # plt.bar(ypos, cnnResults)
    # plt.savefig("CNNAccuracy.pdf")
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], modPreds(preds))



if __name__ == '__main__':
    main()
