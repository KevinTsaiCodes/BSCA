import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main(batch_s, n_epochs, lr) -> None:
    num_classes = 4
    batch_size = batch_s
    num_epochs = n_epochs
    learning_rate = lr

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)

    # Create data loader for the training dataset
    train_dataset = ImageFolder("./dataset/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Use cross-entropy loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Track training accuracy and loss for each epoch
    train_accuracy = []
    train_loss = []

    # Use tqdm to create a progress bar for the epochs
    for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
        model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_accuracy = total_correct / total_samples
        epoch_loss = total_loss / len(train_loader)
        train_accuracy.append(epoch_accuracy)
        train_loss.append(epoch_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Accuracy: {epoch_accuracy:.4f}, Train Loss: {epoch_loss:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_accuracy, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy per Epoch')
    plt.legend()
    plt.savefig('plotting_result/train_accuracy.pdf')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_loss, label='Train Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss per Epoch')
    plt.legend()
    plt.savefig('plotting_result/train_loss.pdf')
    plt.show()


    # Save the model in .pt format
    torch.save(model.state_dict(), 'model/brain_slice_classifier_model.pt')
    print("Model saved as 'brain_slice_classifier_model.pt'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-n", "--n_epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("-l", "--lr", type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    main(args.batch_size, args.n_epochs, args.lr)