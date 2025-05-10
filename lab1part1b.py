########## 209AS Lab1 Problem 1b ################

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

"""Now let's select GPU to train the model by setting the device parameter. Pytorch provides a free GPU for you to use."""

device = 'cuda' if torch.cuda.is_available() else 'cpu' #set the 'device' to GPU if a GPU is detected, otherwise set to CPU
print('Using {} device'.format(device))

"""Define the data transformations and load the CIFAR-10 dataset"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

"""Define a custom simple neural network (LeNet-like). You need to read Pytorch's documentation of nn.Conv2d to understand waht do the input parameters mean. Here we define a simple 5-layer CNN with 2 convolution layers and 2 fully connected layers."""

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        """
        pool size = 2
        first convolution out size = (32 + 2*1 - 3)/2 + 1 = 16
        """

        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2, padding = 1, bias=False) #first channel must set to 3 for this dataset as it has three color channels
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding = 1, bias=False) #this is 32 filter cubes (2nd parameter) with depth 32 (first parameter) of size 3x3
        # self.pool2 = nn.MaxUnpool2d(2,2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding = 1, bias=False)
        # self.pool3 = nn.MaxPool2d(2,2)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64*4*4, 10, bias=False) #(infeatures, outfeatures) --> y = Wx + b, this just automatically caluclates the size of W
        self.bn4 = nn.BatchNorm1d(10)


    def forward(self, x):
        # layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        # layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        # flatten to fully connected
        # print(x.shape)
        x = x.view(-1, 64 * 4 * 4)
        
        x = self.fc1(x)
        x = self.bn4(x)

        return x

net = SimpleNet().to(device) #.to(device) send the define neural network to the specified device

"""Define the loss function and optimizer. Cross entropy loss is typically the default choice for classification problems. Again you can check Pytorch's documentation to see what optimizers you can use (there are plenty of them). Some common choices are SGD and adam."""

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001) #0.001 is the default LR for Adam.

print(net)
"""Train the network. the number of epoch is set to 10 for quicker demonstration. In general you want to train for a bit longer until the network converges."""

net.train()
num_epoch = 12
for epoch in range(num_epoch):
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 200 == 199 or i == len(trainloader)-1:
            # Print more information
            print(f'Epoch [{epoch + 1}/{num_epoch}], Step [{i + 1}/{len(trainloader)}], '
                  f'Loss: {running_loss / 200:.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%, '
                  f'Time: {time.time() - start_time:.2f}s')
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()  # Reset timer


    net.eval()  # Set the model to evaluation mode
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # No need to calculate gradients during validation
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f'Epoch [{epoch + 1}/{num_epoch}], Validation Accuracy: {val_accuracy:.2f}%')

    net.train()  # Set the model back to training mode for the next epoch

print('Finished Training')


"""Test the trained network, typically called 'inference'."""

net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

print(f'8-bit accuracy:{100 * correct / total:.8f}%')
print(f'6-bit accuracy:{100 * correct / total:.6f}%')


"""Finally, save the network"""
torch.save(net.state_dict(), 'prob1b.pth')