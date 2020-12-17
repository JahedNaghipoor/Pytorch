# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 64
learning_rate = 1e-3
batch_size = 64
num_epochs = 2

# create RNN
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = dataloader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = dataloader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters, lr=learning_rate)
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=device).squeeze(1)
        target = target.to(device=device)

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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            score = model(x)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

            print(f'accuracy: {float(num_correct)}/{float(num_samples)*100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

