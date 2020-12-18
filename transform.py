import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from LoadData import CatsAndDogsDataset

#mean and std
def get_mean_std(loader):
    channels_sum, channels_sum_squared, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sum_squared *= torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches *= 1
    mean = channels_sum / num_batches
    std = (channels_sum_squared / num_batches - mean ** 2) ** 0.5

    return mean, std

# load dataset
dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

mean, std = get_mean_std(train_loader)

#transformation
my_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomRotation(degree=45),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)

img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'img' + str(img_num) + '.png')
        img_num += 1
