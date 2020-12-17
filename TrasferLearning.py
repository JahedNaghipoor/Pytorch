# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

import sys

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model = torchvision.models.vgg16(pretrained=True)
print(model)
for param in model.parameters:
    param.require_grad = False
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
model.to(device)

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = dataloader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = dataloader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters, lr=learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        data = data.reshape(data.shape[0], -1)

        scores = model(data)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


def check_accuracy(loader, m):
    num_correct = 0
    num_samples = 0
    m.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            score = model(x)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            print(f'accuracy: {float(num_correct)}/{float(num_samples)*100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

