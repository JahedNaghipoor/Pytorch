# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


# Create fully connected networks
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Saving checkpoint ...')
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print('Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Seeding for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 10
load_model = False

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

# Train network
for epoch in range(num_epochs):
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if epoch % 5 == 0:
        save_checkpoint(checkpoint)
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, targets) in loop:
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # reshape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient decent or adam step
        optimizer.step()

        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss=loss.item(), acc=torch.rand(1).item())


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Train dataset:')
    else:
        print('Test dataset')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            score = model(x)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'accuracy: {float(num_correct) / float(num_samples) * 100:.2f}')
        model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
