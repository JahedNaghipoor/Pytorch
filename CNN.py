# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

        self.initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    # Weight initialization
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rates = [0.1, 0.001, 1e-3, 1e-4]
batch_sizes = [1, 64, 1024]
num_epochs = 1


for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 0
        model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
        writer = SummaryWriter(f'runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}')

        for epoch in range(num_epochs):
            losses = []
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
            for batch_idx, (data, targets) in loop:
                data = data.to(device=device)
                targets = targets.to(device=device)

                # forward
                scores = model(data)
                loss = criterion(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()

                # gradient decent
                optimizer.step()

                mean_loss = sum(losses)/len(losses)
                scheduler.step(mean_loss)

                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_training_acc = float(num_correct)/float(data.shape[0])

                writer.add_scalar('Training Loss', loss, global_step=step)
                writer.add_scalar('Training Accuracy', running_training_acc, global_step=step)
                step += 1


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            score = model(x)
            _, predictions = score.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'accuracy: {float(num_correct)/float(num_samples)*100:.2f}')


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

