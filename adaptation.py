import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from get_datasets import get_data, get_dataloader, get_weak_transforms, get_strong_transforms
from network_pytorch_i3d import InceptionI3d
from torch.nn import DataParallel
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

# Global variables for tracking metrics
target_train_acc_list = []
target_train_loss_list = []
target_val_acc_list = []
target_val_loss_list = []
checkpoint_train_loss_list = []
checkpoint_train_acc_list = []
checkpoint_val_loss_list = []
checkpoint_val_acc_list = []

def set_seed(seed):
    print(f"[ Using Seed : {seed} ]")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_cross_entropy_loss(outputs, labels):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(outputs, labels)
    return loss.item()

def save_checkpoint(epoch, model, optimizer, loss, acc, dataset):
    checkpoint_dir = f'to_{dataset}_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{dataset}_checkpoint_epoch_{epoch}.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.\n")

def plot_metrics(train_list, val_list, title, xlabel, ylabel, filename):
    plt.figure()
    plt.plot(range(1, len(train_list) + 1), train_list, label='Train')
    plt.plot(range(1, len(val_list) + 1), val_list, label='Validation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)


def load_model_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict({key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()})
    return model

def prepare_model(num_classes, in_channels, device, checkpoint_path=None):
    model = InceptionI3d(num_classes=num_classes, in_channels=in_channels)
    model.replace_logits(num_classes=12)  # Assuming 12 classes for this example
    model.to(device)
    if checkpoint_path:
        model = load_model_checkpoint(model, checkpoint_path, device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    return model


def run_inference(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_to_load = 'hmdb51' if args.dataset == 'ucf101' else 'ucf101'
    mode = 'train' if args.val == 0 else 'val'
    checkpoint_path = os.path.join('output_dirs', f'{dataset_to_load}_to_{dataset_to_load}_output_directory/{dataset_to_load}_to_{dataset_to_load}_best_checkpoint.pth')

    # Prepare data loaders
    train_weak_transform = get_weak_transforms(args, mode='train')
    train_strong_transform = get_strong_transforms(args, mode='train')
    val_weak_transform = get_weak_transforms(args, mode='val')
    val_strong_transform = get_strong_transforms(args, mode='val')
    
    train_dataset = get_data((train_weak_transform, train_strong_transform), args, mode='train2', dataset=args.dataset)
    val_dataset = get_data(val_weak_transform, args, mode='val2', dataset=args.dataset)
    
    train_loader = get_dataloader(args, mode='train', dataset=train_dataset)
    val_loader = get_dataloader(args, mode='val', dataset=val_dataset)

    data_loader = train_loader if args.val == 0 else val_loader
    print(f'Passing {"train" if args.val == 0 else "val"} loader')
    
    # Initialize model for inference
    model1 = prepare_model(num_classes=400, in_channels=3, device=device, checkpoint_path=checkpoint_path)
    model1.eval()

    # Initialize model for fine-tuning
    model2 = prepare_model(num_classes=400, in_channels=3, device=device, checkpoint_path=checkpoint_path)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model2.parameters(), weight_decay=1e-7, momentum=0.9, lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    NUM_EPOCHS = 60
    # Initialize max accuracy
    max_acc = 0.0
    max_acc_epoch = 0

    for epoch in range(NUM_EPOCHS):
        model1.eval()
        correct_predictions, total_samples = 0, 0
        video_losses, predicted_video_info = {}, []

        with torch.no_grad():
            for inputs, labels, label_names in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model1(inputs).reshape(inputs.shape[0], -1)
                _, predicted = torch.max(outputs.data, 1)

                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                for path, frame, label, output in zip(label_names, inputs, predicted, outputs):
                    loss = compute_cross_entropy_loss(output.unsqueeze(0), label.unsqueeze(0))
                    video_losses[path] = loss

                    video_info = {
                        'Video_Path': path,
                        'Video_Frames': frame.size(0),
                        'Predicted_Label': label.item()
                    }
                    predicted_video_info.append(video_info)

        val_accuracy = 100.0 * correct_predictions / total_samples
        print(f"\n\n\nValidation Accuracy of {args.dataset} = : {val_accuracy:.4f}")

        # Select and save video information
        predicted_video_info.sort(key=lambda x: x['Predicted_Label'])
        selected_video_info = []
        unique_labels = set(item['Predicted_Label'] for item in predicted_video_info)
        for label in unique_labels:
            label_videos = [item for item in predicted_video_info if item['Predicted_Label'] == label]
            label_videos.sort(key=lambda x: video_losses[x['Video_Path']])
            num_selected = int(0.6 * len(label_videos))
            selected_video_info.extend(label_videos[:num_selected])

        modified_video_info = [{'Video_Path': item['Video_Path'], 'Cross_Entropy_Loss': video_losses[item['Video_Path']], 'Predicted_Label': item['Predicted_Label']}
                               for item in selected_video_info]

        output_dir = 'splits'
        output_file = f'cleanPL_split_{args.dataset}_train.txt'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(output_path):
            os.remove(output_path)

        with open(output_path, 'a') as txtfile:
            for video_info in modified_video_info:
                txtfile.write(f"{video_info['Video_Path']} {video_info['Cross_Entropy_Loss']} {video_info['Predicted_Label']}\n")
        print(f"Modified video information saved to {output_file}")





        print('******************************Fine-Tuning******************************************')
        # Prepare data loaders for fine-tuning
        target_train_loader = get_dataloader(args, mode='train', dataset=get_data((train_weak_transform, train_strong_transform), args, mode='train3', dataset=args.dataset))
        target_val_loader = get_dataloader(args, mode='val', dataset=get_data(val_weak_transform, args, mode='val3', dataset=args.dataset))

        model2.train()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0

        for inputs, labels, label_names in tqdm(target_train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model2(inputs).reshape(inputs.shape[0], -1)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        train_epoch_loss = running_loss / len(target_train_loader.dataset)
        target_train_loss_list.append(train_epoch_loss)
        train_accuracy = 100.0 * correct_predictions / total_samples
        target_train_acc_list.append(train_accuracy)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        scheduler.step()

        model2.eval()
        running_loss, correct_predictions, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels, label_names in tqdm(target_val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model2(inputs).reshape(inputs.shape[0], -1)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        val_epoch_loss = running_loss / len(target_val_loader.dataset)
        target_val_loss_list.append(val_epoch_loss)
        val_accuracy = 100.0 * correct_predictions / total_samples
        target_val_acc_list.append(val_accuracy)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


        output_folder = os.path.join('output_dirs', f'{dataset_to_load}_to_{args.dataset}_output_directory')
        os.makedirs(output_folder, exist_ok=True)

        if val_accuracy > max_acc:
            max_acc = val_accuracy
            max_acc_epoch = epoch + 1
            torch.save(model2.state_dict(), os.path.join(f'{dataset_to_load}_to_{args.dataset}_output_directory', f'{dataset_to_load}_to_{args.dataset}_best_checkpoint.pth'))

        if (epoch + 1) % 3 == 0:
            checkpoint_train_loss_list.append(train_epoch_loss)
            checkpoint_train_acc_list.append(train_accuracy)
            checkpoint_val_loss_list.append(val_epoch_loss)
            checkpoint_val_acc_list.append(val_accuracy)

    # Plot metrics
    plot_metrics(target_train_loss_list, target_val_loss_list, 'Training and Validation Loss', 'Epoch', 'Loss', f'{dataset_to_load}_to_{args.dataset}_loss.png')
    plot_metrics(target_train_acc_list, target_val_acc_list, 'Training and Validation Accuracy', 'Epoch', 'Accuracy', f'{dataset_to_load}_to_{args.dataset}_acc.png')
    plot_metrics(checkpoint_train_loss_list, checkpoint_val_loss_list, 'Training and Validation Loss after every Checkpoints', 'Epoch', 'Loss', f'{dataset_to_load}_to_{args.dataset}_checkpoint_loss.png')
    plot_metrics(checkpoint_train_acc_list, checkpoint_val_acc_list, 'Training and Validation Accuracy after every Checkpoints', 'Epoch', 'Accuracy', f'{dataset_to_load}_to_{args.dataset}_checkpoint_acc.png')

    print("Training complete.")
    print(f'Best accuracy = {max_acc:.4f}, found at epoch = {max_acc_epoch}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--dataset", type=str, default="ucf101", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--val", type=int, default=0, help="0 to finetune or 1 to validate the finetuned result")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()
    run_inference(args)
