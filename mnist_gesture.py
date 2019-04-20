# Kaustav Vats(2016048)

import cv2
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pickle

# Reading Data --------------------
PATH = "./Data/MNIST/"

def ReadData():
    train = []
    train_labels = []
    test = []
    test_labels = []

    with open(PATH + 'sign_mnist_train.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            train_labels.append(row.pop(0))
            train.append(row)
    csvFile.close()

    with open(PATH + 'sign_mnist_test.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            test_labels.append(row.pop(0))
            test.append(row)
    csvFile.close()

    train = np.asarray(train, dtype=np.uint8)
    train_labels = np.asarray(train_labels)

    test = np.asarray(test, dtype=np.uint8)
    test_labels = np.asarray(test_labels)
    print("Train shape: {}".format(train.shape))
    print("Train Labels shape: {}".format(train_labels.shape))

    print("Test shape: {}".format(test.shape))
    print("Test Labels shape: {}".format(test_labels.shape))

    print("Classes: {}".format(np.unique(train_labels).shape))

    return train, train_labels, test, test_labels



# X_Train, Y_Train, X_Test, Y_Test = ReadData()

# np.save(PATH + "X_Train.npy", X_Train)
# np.save(PATH + "Y_Train.npy", Y_Train)
# np.save(PATH + "X_Test.npy", X_Test)
# np.save(PATH + "Y_Test.npy", Y_Test)

X_Train = np.load(PATH + "X_Train.npy")
Y_Train = np.load(PATH + "Y_Train.npy")
X_Test = np.load(PATH + "X_Test.npy")
Y_Test = np.load(PATH + "Y_Test.npy")

print("[+] Data Reading done")

# data Visualization -------------------------

def Visualize(x_data, n_images=5):
    for i in range(n_images):
        image = np.reshape(x_data[i], (28, 28))
        cv2.imwrite(PATH + "temp_" + str(i) + ".png", image)


# Visualize(X_Train)

# Preprocessing ---------------------------------
print("[+] PreProcessing...")



print("[+] PreProcessing Done")
# Features Extraction ---------------------------
print("[+] Features Extraction...")
# Sift Features
def SiftFeatures(x_data, y_data):
    sift = cv2.xfeatures2d.SIFT_create()
    features = []
    labels = []

    for i in range(x_data.shape[0]):
        image = np.reshape(x_data[i], (28, 28))
        _, des = sift.detectAndCompute(image, None)
        if des is None:
            continue
        for d in des:
            features.append(d)
        labels.append(y_data[i])

    features = np.asarray(features)
    labels = np.asarray(labels)

    print("Features shape: {}".format(features.shape))
    print("Labels shape: {}".format(labels.shape))
    return features, labels

def Hog(x_data, y_data, n_images=5):
    features = []
    for i in range(x_data.shape[0]):
        image = np.reshape(x_data[i], (28, 28))
        fd, hog_image = hog(image, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys", visualize=True)
        if i < n_images:
            cv2.imwrite(PATH + "hog_" + str(i) + ".png", hog_image)
        features.append(fd)
    features = np.asarray(features)

    print("Features shape: {}".format(features.shape))
    return features, y_data


NX_Train,  NY_Train = SiftFeatures(X_Train, Y_Train)
# NX_Train,  NY_Train = Hog(X_Train, Y_Train)
# NX_Test,  NY_Test = Hog(X_Test, Y_Test)

# np.save(PATH + "NX_Train.npy", NX_Train)
# np.save(PATH + "NY_Train.npy", NY_Train)
# np.save(PATH + "NX_Test.npy", NX_Test)
# np.save(PATH + "NY_Test.npy", NY_Test)

# NX_Train = np.load(PATH + "NX_Train.npy")
# NY_Train = np.load(PATH + "NY_Train.npy")
# NX_Test = np.load(PATH + "NX_Test.npy")
# NY_Test = np.load(PATH + "NY_Test.npy")

print("[+] Features Extraction Done")

# Unsupervised Learning ----------------------------------------

print("[+] Unsupervised Learning[KMeans]...")

kmeans = KMeans(n_clusters=100, n_jobs=-1)
kmeans.fit(NX_Train)
pickle.dump(kmeans, open(PATH + "Kmean.sav", 'wb'))

# Creating Bag of Visual Words (Creating Vocaboulary)
def Bovw(kmeans, x_data, y_data):
    sift = cv2.xfeatures2d.SIFT_create()
    features = []
    labels = []

    for i in range(x_data.shape[0]):
        image = np.reshape(x_data[i], (28, 28))
        histogram = np.zeros(len(kmeans.cluster_centers_))
        kp, des = sift.detectAndCompute(image, None)
        if des is None:
            continue
        nkp = np.size(kp)
        labels.append(y_data[i])
        for d in des:
            idx = kmeans.predict(np.reshape(d, (1, d.shape[0])))
            histogram[idx] += 1 / nkp

        features.append(histogram)

    features = np.asarray(features)
    labels = np.asarray(labels)

    print("Features shape: {}".format(features.shape))
    print("Labels shape: {}".format(labels.shape))
    return features, labels

NX_Train, NY_Train = Bovw(kmeans, X_Train, Y_Train)
NX_Test, NY_Test = Bovw(kmeans, X_Test, Y_Test)



# np.save(PATH + "NX_Train.npy", NX_Train)
# np.save(PATH + "NY_Train.npy", NY_Train)
# np.save(PATH + "X_Test.npy", X_Train)
# np.save(PATH + "Y_Test.npy", Y_Train)

# NX_Train = np.load(PATH + "NX_Train.npy")
# NY_Train = np.load(PATH + "NY_Train.npy")
# X_Test = np.load(PATH + "X_Test.npy")
# Y_Test = np.load(PATH + "Y_Test.npy")

print("[+] Unsupervised Learning[KMeans] Done")


# Training Classifier ------------------------------------------

print("[+] Training SVM ...")

clf = svm.SVC(gamma='auto', kernel="rbf")
clf.fit(NX_Train, NY_Train)

print("[+] Training SVM done")

print("[+] Testing SVM...")

Accuracy = clf.score(NX_Train, NY_Train)
print("[SVM]Accuracy[Train]:", Accuracy*100)
Accuracy = clf.score(NX_Test, NY_Test)
print("[SVM]Accuracy[Test]:", Accuracy*100)

print("[+] Testing SVM Done")

print("[+] Random Forest Classifier...")

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf.fit(NX_Train, NY_Train)

Accuracy = clf.score(NX_Train, NY_Train)
print("[RFC]Accuracy[Train]:", Accuracy*100)

Accuracy = clf.score(NX_Test, NY_Test)
print("[RFC]Accuracy[Test]:", Accuracy*100)

print("[+] Random Forest Classifier done")
