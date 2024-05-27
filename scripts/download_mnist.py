import os
import argparse
import torchvision
from torchvision import datasets, transforms
from PIL import Image

# Define a function to save the dataset as images
def save_dataset_as_images(dataset, dataset_type, root_dir):
    for index, (image, label) in enumerate(dataset):
        label_dir = os.path.join(root_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        image_path = os.path.join(label_dir, f'{dataset_type}_{index}.png')
        image_pil = transforms.ToPILImage()(image)
        image_pil.save(image_path)


def main(root_dir):
    # Define a transform to convert the data to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download the MNIST training and test datasets
    train_dataset = datasets.MNIST(root=root_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)

    # Save training and test datasets as images in the same root directory
    save_dataset_as_images(train_dataset, 'train', root_dir)
    save_dataset_as_images(test_dataset, 'test', root_dir)

    print("MNIST dataset saved as images successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MNIST dataset and save as images')
    parser.add_argument('root_dir', type=str, help='Root directory to save the MNIST dataset images')
    
    args = parser.parse_args()
    main(args.root_dir)
