import torch
import numpy as np
from glob import glob
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize


torch.manual_seed(0)
tt = Compose([Resize((24, 24)), ToTensor()])


class Sentinel2Dataset:
    def __init__(self, folder_path, train=True, accumulate=False, **kwargs):
        self.folder_path = folder_path
        self.train = train
        self.accumulate = accumulate

    def __len__(self):
        return len(glob(f"{self.folder_path}/images/*npy"))

    def __getitem__(self, idx):
        image_path = f"{self.folder_path}/images/{idx}.npy"
        image = np.load(image_path) / 10000.0
        if self.accumulate:
            image = self.accumulate_image(image)
        image = Image.fromarray(image.astype("float64"))
        if self.train:
            mask_path = f"{self.folder_path}/masks/{idx}.npy"
            mask = np.load(mask_path)
            mask = Image.fromarray(mask.astype("int"))
            return tt(image), tt(mask)
        else:
            return tt(image)

    def accumulate_image(self, image):
        return np.sum(image, axis=2) / image.shape[2]


def test():
    ds = Sentinel2Dataset("data/train", accumulate=True)
    image, mask = ds[0]

    print(image.shape, mask.shape)


if __name__ == "__main__":
    test()
