import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DrawingTutorialDataset(Dataset):
    def __init__(self, json_path, images_dir, n_frames=6, image_size=256, n_noisy=99):
        self.images_dir = images_dir
        self.n_frames   = n_frames
        self.image_size = image_size
        self.n_noisy    = n_noisy

        with open(json_path) as f:
            tutorials = json.load(f)

        self.samples = []
        for tut in tutorials:
            self.samples.append({"tutorial": tut, "noise_std": 0})
            for i in range(n_noisy):
                noise_std = np.random.uniform(5, 30)
                self.samples.append({"tutorial": tut, "noise_std": noise_std})

        print(f"Dataset: {len(tutorials)} tutoriale → {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _load_and_transform(self, image_path, noise_std):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))

        img_array = np.array(img).astype(np.float32)

        if noise_std > 0:
            noise     = np.random.normal(0, noise_std, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

        img = Image.fromarray(img_array)
        img = T.ToTensor()(img)
        img = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        return img

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        tutorial  = sample["tutorial"]
        noise_std = sample["noise_std"]

        frames = []
        for step in tutorial["steps"]:
            img_path = os.path.join(self.images_dir, step["drawing_image"])
            img      = self._load_and_transform(img_path, noise_std)
            frames.append(img)

        pixel_values = torch.stack(frames)
        text         = tutorial["title"]

        return {
            "pixel_values": pixel_values,
            "text":         text
        }