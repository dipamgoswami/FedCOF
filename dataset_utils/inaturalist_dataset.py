import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class INaturalistDataset(Dataset):
    def __init__(self, images_root_dir, csv_file, transform=None):
        """
        Args:
            images_root_dir (str): Root directory of images (e.g., 'train_val_images' or 'test_images')
            csv_file (str): Path to the CSV file containing 'image_id', 'label', and optionally 'user' columns
            transform (callable, optional): Optional transform to be applied on an image
        """
        self.images_root_dir = images_root_dir
        self.transform = transform

        # Read the CSV file
        self.data = pd.read_csv(csv_file)
        # Ensure 'image_id' is of type string
        self.data['image_id'] = self.data['image_id'].astype(str)

        # Check if 'user' column exists
        self.has_users = 'user_id' in self.data.columns

        # Build a mapping from image_id to image_path
        self.image_id_to_path = self._create_image_id_to_path_mapping()

        # Filter the dataset to only include images that exist
        self.data = self.data[self.data['image_id'].isin(self.image_id_to_path.keys())].reset_index(drop=True)

    def _create_image_id_to_path_mapping(self):
        """
        Creates a mapping from image_id to the full image path by walking through the image directory.
        """
        image_id_to_path = {}
        for root, _, files in os.walk(self.images_root_dir):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff')):
                    # Extract image_id without extension
                    image_id = os.path.splitext(file)[0]
                    image_path = os.path.join(root, file)
                    image_id_to_path[image_id] = image_path
        return image_id_to_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image_id, label, and user (if available) for the given index
        row = self.data.iloc[idx]
        image_id = str(row['image_id'])
        label = row['class']

        # Get the image path from the mapping
        image_path = self.image_id_to_path.get(image_id)
        if image_path is None:
            # Handle the case where the image_id is not found
            raise FileNotFoundError(f"Image with ID {image_id} not found in directory {self.images_root_dir}")

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        if self.has_users:
            user = row['user_id']
            return  image, label, user
        else:
            return image, label