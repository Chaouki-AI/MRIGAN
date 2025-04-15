import io
import glob
import torch
import random
import pydicom
import numpy as np
from PIL import Image
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, shape = (512, 512), scales = [1, 8], compress = [10, 80], blur = [0.2, 2], RGB = True):
        self.image_paths = image_paths
        # Define a basic transformation for HR images:
        
        self.basic_transform = transforms.Compose([transforms.ToTensor()])
        self.type_img = "RGB" if RGB else "L"
        self.shape    = shape
        self.scales   = scales
        self.compress = compress
        self.blur     = blur
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if '.jpg' in image_path : 
            image      = Image.open(image_path)
        else : 
            image = self.read_dcm(image_path)
        image = image.convert(self.type_img).resize(self.shape)
        hr_tensor  = self.basic(image)
        lr_tensor  = self.transform(image)
        return {"HR": hr_tensor, "LR": lr_tensor}
        
    def read_dcm(self, path):
        ds = pydicom.dcmread(path).pixel_array.astype(np.float32)
        ds /= ds.max()
        ds *= 255
        ds = np.concatenate([np.expand_dims(ds, 2), np.expand_dims(ds, 2), np.expand_dims(ds, 2)], -1)
        ds = ds.astype(np.uint8)
        image = Image.fromarray(ds)
        return image

    def transform(self, img):
        """
        Applies random degradation to the input PIL image to simulate a low-resolution (LR) input.
        The degradation includes random Gaussian blur, random downsampling, and JPEG compression artifacts.
        """
        # 1. Apply random Gaussian Blur
        blur_radius = random.uniform(self.blur[0], self.blur[1])
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 2. Choose a random scale factor for downsampling (e.g., between 2 and 4)
        scale = random.randint(self.scales[0], self.scales[1])
        w, h = blurred.size
        lr_size = (w // scale, h // scale)
        downsampled = blurred.resize(lr_size, Image.BICUBIC)

        # 3. Simulate JPEG compression artifacts by saving and reloading the image in-memory
        quality = random.randint(self.compress[0], self.compress[1])
        buffer = io.BytesIO()
        downsampled.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        degraded = Image.open(buffer).convert(self.type_img)
        degraded = degraded.resize(self.shape)

        # 4. Convert the degraded PIL image to tensor (no normalization here, as LR may serve as input to the network)
        lr_tensor = transforms.ToTensor()(degraded)
        return lr_tensor

    def basic(self, img):
        """
        Converts the input PIL image (HR) to a tensor and applies normalization.
        """
        return self.basic_transform(img)

