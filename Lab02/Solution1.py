import os
from torch.utils import data
from torchvision import datasets, transforms

# Downloaded from https://www.kaggle.com/datasets/sanadalali/animal-categories-90-masters-of-survival
images_dir = 'animal-categories-90-masters-of-survival/animals/animals/animals'

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to 224x224
    transforms.RandomRotation(30), # Randomly rotate images by 30 degrees
    transforms.RandomCrop(200), # Randomly crop images to 200x200
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), # Randomly change brightness, contrast, saturation, and hue
    transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
])

# Define the dataset
dataset = datasets.ImageFolder(root=images_dir, transform=transform)

subset_size = int(len(dataset) * 0.2) # 20% of the dataset
dataset, _ = data.random_split(dataset, [subset_size, len(dataset) - subset_size]) # Randomly select 20% of the dataset

# Save the augmented images
output_dir = 'augmented_dataset'
os.makedirs(output_dir, exist_ok=True)

for i, (image, label) in enumerate(dataset):
    label_dir = os.path.join(output_dir, dataset.dataset.classes[label])
    os.makedirs(label_dir, exist_ok=True)
    image.save(os.path.join(label_dir, f'augmented_{i}.jpg'))
    print(f"Saved augmented image {i} to {label_dir}")
print(f"Augmented dataset saved to {output_dir}")
