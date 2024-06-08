import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from get_datasets import get_data, get_dataloader, get_weak_transforms, get_strong_transforms
from network_pytorch_i3d import InceptionI3d
from torch.nn import DataParallel
import argparse
import os

def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict({key.replace("module.", ""): value for key, value in checkpoint['model_state_dict'].items()})
    return model

def get_dataset_transforms(args, mode):
    weak_transforms = get_weak_transforms(args, mode=mode)
    strong_transforms = get_strong_transforms(args, mode='train')
    if mode == 'val':
        return weak_transforms
    else:
        return weak_transforms, strong_transforms

def get_dataloaders(args, dataset, mode):
    transforms = get_dataset_transforms(args, mode)
    dataset_instance = get_data(transforms, args, mode=mode+'1', dataset=dataset)
    return get_dataloader(args, mode=mode, dataset=dataset_instance)

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_to_load = 'hmdb51' if args.dataset == 'ucf101' else 'ucf101'
    mode = 'train' if args.pseudo_label else 'val'

    if mode == 'train':
        checkpoint_name = f'{dataset_to_load}_to_{dataset_to_load}_best_checkpoint.pth'
        checkpoint_path = os.path.join('output_dirs', f'{dataset_to_load}_to_{dataset_to_load}_output_directory', checkpoint_name)
    else:
        checkpoint_name = f'{args.dataset}_to_{args.dataset}_best_checkpoint.pth'
        checkpoint_path = os.path.join('output_dirs', f'{args.dataset}_to_{args.dataset}_output_directory', checkpoint_name)

    # Initialize your model
    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(num_classes=12)  # Call the replace_logits method
    model.to(device)

    if args.pseudo_label:
        print('Passing train loader')
    else:
        print('Passing val loader')

    print("==> Loading pretrained weights from {}".format(checkpoint_path))
    model = load_checkpoint(model, checkpoint_path, device)

    if torch.cuda.device_count() > 1:
        print('\nNo of GPUs using: ', torch.cuda.device_count())
        model = DataParallel(model)

    model.eval()
    dataloader = get_dataloaders(args, args.dataset, mode)

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    class_correct = [0.0] * 12
    class_total = [0.0] * 12

    predicted_video_info = []

    with torch.no_grad():
        for inputs, labels, label_names in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = torch.reshape(outputs, [outputs.shape[0], outputs.shape[1]])
            _, predicted = torch.max(outputs.data, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            # Append video information
            for path, frame, label in zip(label_names, inputs, predicted):
                video_info = {
                    'Video_Path': path,
                    'Video_Frames': frame.size(0),  # Assuming frame is a tensor
                    'Predicted_Label': label.item()
                }
                predicted_video_info.append(video_info)

    val_accuracy = 100.0 * correct_predictions / total_samples
    print(f"Validation Accuracy of {args.dataset} = : {val_accuracy:.4f}\n")

    for i in range(12):
        print('Accuracy of %5s : %2d %%' % (label_names[i], 100 * class_correct[i] / class_total[i]))
    print('\n')




    # Define the folder path where the .txt file will be saved
    output_folder = 'splits'
    os.makedirs(output_folder, exist_ok=True)

    # Save the predicted video information to a TXT file
    output_file = os.path.join(output_folder, f'PL_split_{args.dataset}_train.txt' if args.pseudo_label else f'PL_split_{args.dataset}_val.txt')


    # Save the predicted video information to a TXT file
    output_file = f'PL_split_{args.dataset}_train.txt' if args.pseudo_label else f'PL_split_{args.dataset}_val.txt'
    with open(output_file, 'w') as txtfile:
        for video_info in predicted_video_info:
            txtfile.write(f"{video_info['Video_Path']}")
            txtfile.write(f" {video_info['Video_Frames']}")
            txtfile.write(f" {video_info['Predicted_Label']}")
            txtfile.write("\n")
    print(f"Predicted video information saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--dataset", type=str, default="ucf101", help="Dataset name")
    parser.add_argument("--pseudo_label", action='store_true', help="Generate pseudo labels")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")

    args = parser.parse_args()
    run_inference(args)
