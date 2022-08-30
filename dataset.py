from torch.utils.data import Dataset
import glob
import torch
import cv2
import pdb

class MRIDataset(Dataset):
    def __init__(self, images_path, masks_path, device='cuda') -> None:
        # Save the images and masks paths
        self.images_path = images_path
        self.masks_path = masks_path
        self.device = device

        # Extract the full paths of all images and their corresponding masks
        self.images = self.extract_filepaths(images_path)
        self.masks = self.extract_filepaths(masks_path)

    def extract_filepaths(self, files_path):
        files = list()
        for file in glob.glob(files_path + "/*.tif"):
            files.append(file)
        return files

    def __getitem__(self, index):
        def load_image(image_path, grayscale):
            """NEED TO RESIZE THE IMAGES TO THE SAME SIZE"""
            if grayscale:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image_path)
                img = img[...,::-1] # the image is loaded as a BGR -- convert it to RGB
            img = torch.tensor(img.copy(), device=self.device, dtype=torch.float)
            return img
        
        image_path = self.images[index]
        mask_path = self.masks[index]

        image = load_image(image_path, grayscale=False)
        mask = load_image(mask_path, grayscale=True)

        return (image, mask)
    
    def __len__(self):
        assert len(self.images) == len(self.masks)
        return len(self.images)