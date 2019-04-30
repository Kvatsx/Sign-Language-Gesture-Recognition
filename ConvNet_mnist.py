'''
 * @Author: Kaustav Vats 
 * @Roll-Number: 2016048 
'''
# Ref: [ConvNet] https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Data Reading ---------------------------------
PATH = "./Data/MNIST/"

NClass = 24

def ReadData():
    train = []
    test = []

    with open(PATH + 'sign_mnist_train.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            label = int(row.pop(0))
            row = np.asarray(row)
            row = np.reshape(row, (1, 28, 28))
            train.append((torch.from_numpy(row.astype(np.float64)), label))
    csvFile.close()

    with open(PATH + 'sign_mnist_test.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        count = 0
        for row in reader:
            if count == 0:
                count += 1
                continue
            label = int(row.pop(0))
            row = np.asarray(row)
            row = np.reshape(row, (1, 28, 28))
            test.append((torch.from_numpy(row.astype(np.float64)), label))            

    csvFile.close()

    print("Train shape: {}".format(len(train)))
    print("Test shape: {}".format(len(test)))

    return train, test

Train, Test = ReadData()

# np.save(PATH + "Train.npy", Train)
# np.save(PATH + "Test.npy", Test)

# Train = np.load(PATH + "Train.npy")
# Test = np.load(PATH + "Test.npy")

# print("Train.shape: {}".format(Train.shape))
# print("Test.shape: {}".format(Test.shape))
# print(Train[:, 1])
# NClass = np.unique(Train[:][1]).shape(0)
# print("Number of Classes: ".format(NClass))

print("[+] Data Reading done")
# Convolution Neural Network Hyperparameter ---------------------------------------
num_epochs = 30
batch_size = 100
learning_rate = 0.001

# Data Loader ----------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=Train, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=Test, batch_size=batch_size, shuffle=False)

print("[+] Data Loader ready")

# CNN Class --------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))
print("Device Name:", torch.cuda.get_device_name(0))

class ConvNet(nn.Module):
    def __init__(self, num_classes=24):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(7*7*32, num_classes)
        # self.soft = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        # out = self.soft(out)
        return out

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images.float())
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(),PATH + 'model.ckpt')



