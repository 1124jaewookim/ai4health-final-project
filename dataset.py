import os
import torch
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = True

def open_image(path):
    return Image.open(path).convert("RGB")

class Ai4healthDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        pass
        # Get path of the train images and labels from the train folder in chest_xray
        # Use glob library to get the path of the images and labels
        
        self.lable_list = ['normal', 'bacterial', 'viral']
        self.base_transform = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.train_images_n = glob(os.path.join(f"chest_xray/{mode}/NORMAL/*.jpeg"))
        self.train_images_n_labels = [0 for _ in range(len(self.train_images_n))]
        
        self.train_images_p = glob(os.path.join(f"chest_xray/{mode}/PNEUMONIA/*.jpeg"))
        self.train_images_p_b = []
        self.train_images_p_v = []
        for path in self.train_images_p:
            if 'virus' in path:
                self.train_images_p_v.append(path)
            else:
                self.train_images_p_b.append(path)
        print(f'############## {mode} ##############')
        print('Normal: ', len(self.train_images_n))
        print("Bacterial: ", len(self.train_images_p_b))
        print("Viral: ", len(self.train_images_p_v))
        
        self.train_images_p_b_labels = [1 for _ in range(len(self.train_images_p_b))]
        self.train_images_p_v_labels = [2 for _ in range(len(self.train_images_p_v))]
        
        self.train_x = self.train_images_n + self.train_images_p_b + self.train_images_p_v
        self.train_y = self.train_images_n_labels + self.train_images_p_b_labels + self.train_images_p_v_labels

        
    def __getitem__(self, idx):
        image_path = self.train_x[idx]
        img = open_image(image_path)
        
        img = self.base_transform(img)
        label = self.train_y[idx]
        
        return img, label, idx
    
    
    def __len__(self):
        return len(self.train_x)
    

test = Ai4healthDataset(mode='train')

    