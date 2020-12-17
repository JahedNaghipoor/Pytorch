import torch
import torchvision.transforms as transforms
from torchviosion.utils import save_image
from LoadData import CatsAndDogsDataset

# transformation
my_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomRotation(degree=45),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ]
)
dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized', transform=my_transforms)
train_loader = dataloader(dataset=dataset, batch_size=batch_size, shuffle=True)

img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'img' + str(img_num) + '.png')
        img_num *= 1

