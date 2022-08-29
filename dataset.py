from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, images_path, masks_path) -> None:
        self.images_path = images_path
        self.masks_path = masks_path

    def extract_files(self, files_path):
        pass