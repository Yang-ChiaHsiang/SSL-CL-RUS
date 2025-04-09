import cv2
import os
import math
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from dataset.transform import image_only_transform, shared_transform, crop, hflip, normalize, resize, blur, cutout

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None, colormap=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.colormap = colormap
        
        self.pseudo_mask_path = pseudo_mask_path
        
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229]),
        ])

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids
    
        else:
            if mode == 'test':
                id_path = 'dataset/splits/%s/test.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train' or mode == 'val':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    @staticmethod
    def encode_segmap(mask, colormap):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(colormap):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask
        
    def __getitem__(self, item):
        id = self.ids[item]
        img_name = os.path.join(self.root, id.split(' ')[0])
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        # img = np.stack((img,)*3, axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        
        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids) or self.mode == 'val' or self.mode == 'label' or self.mode == 'test':
            mask_name = os.path.join(self.root, id.split(' ')[1])
        else:
            mask_name = os.path.basename(id.split(' ')[1])
            mask_name = os.path.join(self.pseudo_mask_path, mask_name)
            
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        
        if self.mode == 'train' or self.mode == 'semi_train':
            sample = shared_transform(image=img, mask = mask)
            img = sample['image']        
            mask = sample['mask']
            
            sample = image_only_transform(image=img)
            img = sample['image']        
            
        # strong augmentation on unlabeled images ST++ 原始論文中的 strong augmentation，輸入需要 PIL 影像。
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            # CV2 to PIL 
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)
            
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)
            
            # PIL to CV2
            img = np.array(img)  # Convert from PIL to NumPy array
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from RGB to BGR
            mask = np.array(mask)  # Convert from PIL to NumPy array
             
        mask = self.encode_segmap(mask, self.colormap)
        img = self.image_transform(img)    
        
        # return
        if self.mode == 'label':
            return img, mask, id
        
        return img, mask

    def __len__(self):
        return len(self.ids)
