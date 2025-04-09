import math
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def augment(imgs):
    """
    INPUT: numpy array of shape num_samples*3*img_size*img_size [dtype = float32]
    OUTPUT: numpy array of same shape as input with augmented images
    """
    
    # Define our sequence of augmentation steps that will be applied to every image.
    aug = A.Compose(
        [
            A.HorizontalFlip(p=0.7),  # horizontally flip 70% of all images
            A.OneOf([
                A.GaussNoise(var_limit=(0.0, 0.05*255), p=0.5),  # Add gaussian noise
                A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=0.5)
            ], p=0.5),
            A.RandomBrightnessContrast(p=0.5),  # color jitter the image
            A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=0.5),
            A.CLAHE(p=0.5),
            A.CropAndPad(percent=(-0.2, 0.2), p=0.5),  # crop some of the images by 0-20% of their height/width
            A.ToGray(p=0.5),  # Convert each image to grayscale with 50% probability
        ]
    )

    augmented_imgs = []
    for img in imgs:
        img = (img * 255).astype(np.uint8)
        augmented = aug(image=img)
        augmented_img = augmented['image']
        augmented_img = (augmented_img / 255.).astype(np.float32)
        augmented_imgs.append(augmented_img)
    
    return np.array(augmented_imgs)

def batch_augment(list_imgs, N):
    """
    INPUT: 
    list_imgs =list of (numpy array of shape num_classes*3*img_size*img_size [dtype = float32])
    N = (int) total number of samples in a batch
    OUTPUT = list of numpy array of same shape as input with augmented images
    """
    num_classes_in_batch = 0
    for imgs in list_imgs:
      if imgs is not None:
          num_classes_in_batch+=1
    
    imgs_per_class= int(math.ceil(N/num_classes_in_batch))
    out_list =[]
    for imgs in list_imgs:
        if imgs is not None:
            if imgs.shape[0]>=imgs_per_class:
              np.random.shuffle(imgs)
              imgs = imgs[:imgs_per_class,:,:,:]
              out_list.append(imgs)
            else :
                imgs_in_cls =int(imgs.shape[0])
                num_augs = int(math.ceil(imgs_per_class/imgs_in_cls))-1
                imgs_to_append = imgs
                for i in range(num_augs):
                    np.random.shuffle(imgs)
                    aug_imgs = augment(imgs.transpose(0,2,3,1))
                    imgs_to_append=np.vstack((imgs_to_append, aug_imgs.transpose(0,3,1,2)))
                imgs_to_append = imgs_to_append[:imgs_per_class,:,:,:]
                out_list.append(imgs_to_append)   
        else :
            out_list.append(None)

    return out_list
