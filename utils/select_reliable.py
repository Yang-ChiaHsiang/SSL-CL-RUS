import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset
from utils.DICELOSS import DiceLoss
import torch.nn.functional as F
import torchvision.transforms as transforms


def softmax_mse_loss(student_outputs, teacher_outputs):
    student_softmax = torch.nn.functional.softmax(student_outputs, dim=1)
    teacher_softmax = torch.nn.functional.softmax(teacher_outputs, dim=1)
    return torch.mean((student_softmax - teacher_softmax) ** 2, dim=1)


def consistency_regularization_CELoss(model, teacher_model, imgs):
    # Define strong and weak augmentations
    strong_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.5)
        # Add more morphological transformations if needed
    ])

    weak_aug = transforms.Compose([
        transforms.RandomRotation(degrees=(0, 30)),  # Random rotation within 0-30 degrees
        transforms.RandomCrop(size=(imgs.shape[2], imgs.shape[3]))  # Random crop, set size as needed
        # Add more augmentations if needed
    ])

    # Apply augmentations to the images
    weak_aug_imgs = weak_aug(imgs)
    strong_aug_imgs = strong_aug(imgs)
    
    # Get predictions from the model and teacher model
    model_preds = model(strong_aug_imgs)
    teacher_preds = teacher_model(weak_aug_imgs)
    
    # Use softmax on both predictions to get probability distributions
    model_probs = F.softmax(model_preds, dim=1)
    teacher_probs = F.softmax(teacher_preds, dim=1)
    
    # Flatten the tensors to match the input requirements of F.cross_entropy
    model_probs = model_probs.permute(0, 2, 3, 1).reshape(-1, model_probs.shape[1])
    teacher_probs = teacher_probs.permute(0, 2, 3, 1).reshape(-1, teacher_probs.shape[1])

    # Compute cross-entropy loss
    consistency_loss = torch.mean(torch.sum(-teacher_probs * torch.log(model_probs + 1e-10), dim=-1))

    return consistency_loss

# def consistency_regularization_CELoss(model, teacher_model, imgs):
#     # 定義強弱增強
#     strong_aug = transforms.Compose([
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=(0, 30)),
#         transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
#     ])
    
#     weak_aug = transforms.Compose([
#         transforms.RandomRotation(degrees=(0, 15)),
#         transforms.RandomCrop(size=(imgs.shape[2], imgs.shape[3])),
#     ])
    
#     # 應用增強到影像
#     weak_aug_imgs = torch.stack([weak_aug(img) for img in imgs])
#     strong_aug_imgs = torch.stack([strong_aug(img) for img in imgs])
    
#     # 模型預測
#     model_preds = model(strong_aug_imgs)
#     teacher_preds = teacher_model(weak_aug_imgs)
    
#     # 獲取概率分佈
#     model_probs = F.softmax(model_preds, dim=1)
#     teacher_probs = F.softmax(teacher_preds, dim=1)
    
#     # 計算 KL 散度損失
#     consistency_loss = F.kl_div(
#         input=torch.log(model_probs + 1e-10),  # Logits需取對數
#         target=teacher_probs,                 # 教師模型作為目標
#         reduction='batchmean'                 # 平均每個樣本的損失
#     )
    
#     return consistency_loss

def select_reliable(model, teacher_model, data_loader, num_classes, threshold=0.1, device='cuda'):
    model.eval()
    teacher_model.eval()
    
    model.module.contrast = False
    teacher_model.module.contrast = False

    reliable_images = []
    reliable_outputs = []
    remaining_images = []

    tbar = tqdm(data_loader)

    with torch.no_grad():
        for imgs in tbar:
            imgs = imgs.to(device)
            noise = torch.clamp(torch.randn_like(imgs) * 0.05, -0.1, 0.1)  # 調整噪聲大小
            ema_inputs = imgs + noise

            student_outputs = model(imgs)
            teacher_outputs = teacher_model(ema_inputs)
            
            # 計算基於 softmax 的 MSE 損失
            consistency_loss = softmax_mse_loss(student_outputs, teacher_outputs)
            
            student_outputs = student_outputs.argmax(dim=1)

            for img, output, loss in zip(imgs, student_outputs, consistency_loss):
                if loss.mean().item() < threshold:
                    reliable_images.append(img.cpu())
                    reliable_outputs.append(output.cpu())
                else:
                    remaining_images.append(img.cpu())

    reliable_images_tensor = torch.stack(reliable_images)
    reliable_outputs_tensor = torch.stack(reliable_outputs)
    reliable_dataset = TensorDataset(reliable_images_tensor, reliable_outputs_tensor)
    remaining_images_tensor = torch.stack(remaining_images)
    remaining_dataset = TensorDataset(remaining_images_tensor)
    
    return reliable_dataset, remaining_dataset


def Label(model, data_loader, num_classes, device='cuda'):
    tbar = tqdm(data_loader)
    images = []
    outputs = []
    
    for batch in tbar:
        print(f"Batch type: {type(batch)}")  # 打印批次的類型
        if isinstance(batch, list):
            print(f"Batch contains {len(batch)} elements")
            imgs = batch[0]  # 假設圖像數據在第一個位置
        elif isinstance(batch, torch.Tensor):
            imgs = batch
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")
        
        imgs = imgs.to(device)
        output = model(imgs)
        output = output.argmax(dim=1)
        
        for img, output in zip(imgs, output):
            images.append(img.cpu().numpy())
            outputs.append(output.cpu().numpy())
            
    images_tensor = torch.tensor(images)
    outputs_tensor = torch.tensor(outputs)
    
    return TensorDataset(images_tensor, outputs_tensor)

