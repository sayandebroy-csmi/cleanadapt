import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
import datetime
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.data import DataLoader
from get_datasets import get_data, get_dataloader, get_weak_transforms, get_strong_transforms
from network_pytorch_i3d import InceptionI3d
from tqdm import tqdm
from torch.nn import DataParallel

def set_seed(seed):
    print(f"[ Using Seed : {seed} ]")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_metrics(train_list, val_list, title, xlabel, ylabel, filename, num_epochs):
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_list, label='Train')
    plt.plot(range(1, num_epochs + 1), val_list, label='Validation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

def initialize_model(num_classes, pretrained_weights_path, device):
    model = InceptionI3d(num_classes=400, in_channels=3)
    pretrained_dict = torch.load(pretrained_weights_path)
    model.load_state_dict(pretrained_dict)
    model.replace_logits(num_classes=num_classes)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f'\nUsing {torch.cuda.device_count()} GPUs')
        model = DataParallel(model)
    return model

def save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, output_dir, dataset_name):
    checkpoint_path = os.path.join(output_dir, f'{dataset_name}_to_{dataset_name}_best_checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_accuracy': best_accuracy
    }, checkpoint_path)



def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.to(device)
    return start_epoch, best_accuracy

def log_training_progress(output_dir, dataset_name, epoch, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy):
    log_entry = f"{datetime.datetime.now()} - Epoch {epoch + 1}/{num_epochs}: "
    log_entry += f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    log_file_path = os.path.join(output_dir, f'{args.dataset_name}_to_{args.dataset_name}_training_log.txt')
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_entry + '\n')

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels, _ in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.reshape([outputs.shape[0], outputs.shape[1]])
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    train_loss = running_loss / total_samples
    train_accuracy = 100.0 * correct_predictions / total_samples
    return train_loss, train_accuracy

def validate(model, val_loader, criterion, device, num_classes=12):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    class_correct = [0.0] * num_classes
    class_total = [0.0] * num_classes

    with torch.no_grad():
        for inputs, labels, label_names in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.reshape([outputs.shape[0], outputs.shape[1]])
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    val_loss = running_loss / total_samples
    val_accuracy = 100.0 * correct_predictions / total_samples
    class_accuracy = [100 * c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    return val_loss, val_accuracy, class_accuracy

def train(args):
    set_seed(args.seed)

    # Hyperparameters
    config = {
        'learning_rate': 0.01,
        'num_epochs': 40,
        'lr_decay_factor': 0.1,
        'lr_decay_epochs': [10, 20],
        'num_classes': 12,
        'pretrained_weights_path': 'pretrained_weights/rgb_imagenet.pt'
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join('output_dirs', f'{args.dataset_name}_to_{args.dataset_name}_output_directory')
    os.makedirs(output_dir, exist_ok=True)


    # Data Transforms and Loaders
    train_transforms = (get_weak_transforms(args, mode='train'), get_strong_transforms(args, mode='train'))
    val_transforms = get_weak_transforms(args, mode='val')
    train_dataset = get_data(train_transforms, args, mode='train1', dataset=args.dataset_name)
    val_dataset = get_data(val_transforms, args, mode='val1', dataset=args.dataset_name)
    train_loader = get_dataloader(args, mode='train', dataset=train_dataset)
    val_loader = get_dataloader(args, mode='val', dataset=val_dataset)

    model = initialize_model(config['num_classes'], config['pretrained_weights_path'], device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=1e-7, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['lr_decay_epochs'], gamma=config['lr_decay_factor'])

    start_epoch = 0
    best_accuracy = 0.0

    if args.resume_checkpoint:
        start_epoch, best_accuracy = load_checkpoint(model, optimizer, scheduler, args.resume_checkpoint, device)
        print(f"Resuming training from epoch {start_epoch + 1} with best accuracy {best_accuracy:.4f}")

    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

    for epoch in range(start_epoch, config['num_epochs']):
        print(f'\nEpoch {epoch + 1}/{config["num_epochs"]}\n' + '-' * 50)
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy, class_accuracy = validate(model, val_loader, criterion, device, config['num_classes'])
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        for i, acc in enumerate(class_accuracy):
            print(f'Accuracy of class {i}: {acc:.2f}%')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch + 1
            save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, output_dir, args.dataset_name)

        log_training_progress(output_dir, args.dataset_name, epoch, config['num_epochs'], train_loss, train_accuracy, val_loss, val_accuracy)
        scheduler.step()

    plot_metrics(train_loss_list, val_loss_list, 'Training and Validation Loss', 'Epoch', 'Loss', f'{output_dir}/loss.png', config['num_epochs'])
    plot_metrics(train_acc_list, val_acc_list, 'Training and Validation Accuracy', 'Epoch', 'Accuracy', f'{output_dir}/accuracy.png', config['num_epochs'])

    print(f"Training complete.\nBest accuracy = {best_accuracy:.4f}, found at epoch = {best_epoch}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--dataset_name", type=str, default="ucf101", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to checkpoint to resume training from", default=None)
    args = parser.parse_args()
    train(args)
