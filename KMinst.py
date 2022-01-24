import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# I was forced to run on CPU but if you have GPU support add it by replacing the current one with this: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#hyper-parameters
num_epoch = 35
learning_rate = 0.002
input_size = 784
num_classes = 10
hidden_size = 520
batch_size = 200

#downloading and loading data 
train_dataset = torchvision.datasets.KMNIST(root='./dataKM', train=True, download=True, transform=transforms.ToTensor())

test_dataset = torchvision.datasets.KMNIST(root='./dataKM', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#displays the images 
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

#Neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out

    

#model
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

#loss
l = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_t_s = len(train_loader)

#training loop
for epoch in range(num_epoch):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        ot = model(images)
        loss = l(ot, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 100 == 0:
            print(f' epoch : {epoch+1}/{num_epoch}, step : {i+1}/{n_t_s}, loss : {loss.item():.4f}')

#test model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    




        
