import numpy as np
from PIL import Image
from utils.DICELOSS import DiceLoss
import torch.nn.functional as F
import torch

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def calculate_metrics_and_confusion_matrix(pred, target, threshold=0.5):
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    target_one_hot = target_one_hot[:, 1:, :, :]

    pred = pred [:, 1:, :, :]
    pred = torch.sigmoid(pred)
    
    # Apply threshold to predictions
    pred_bin = (pred > threshold).float()
    
    TP = ((pred_bin == 1) & (target_one_hot == 1)).sum().item()
    FP = ((pred_bin == 1) & (target_one_hot == 0)).sum().item()
    FN = ((pred_bin == 0) & (target_one_hot == 1)).sum().item()
    TN = ((pred_bin == 0) & (target_one_hot == 0)).sum().item()
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    return sensitivity, specificity, TP, TN, FP, FN
    
    
class DiceCoefficient:
    def __init__(self):
        self.dice_scores = []

    def add_batch(self, predictions, gts):
        # 初始化 DiceLoss
        dice_loss_fn = DiceLoss()
        
        # 計算 Dice loss
        dice_loss = dice_loss_fn(predictions, gts)
        
        # 計算 Dice coefficient
        dice = 1 - dice_loss
        
        # 添加到列表中
        self.dice_scores.append(dice.item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.dice_scores)).item()

class MeanIOU:
    def __init__(self):
        self.miou_scores = []

    def add_batch(self, predictions, gts):
        
        gts_one_hot = F.one_hot(gts, num_classes=predictions.shape[1]).permute(0, 3, 1, 2).float()
        gts_one_hot = gts_one_hot[:,1:,:,:]
        
        # If your model contains a sigmoid activation layer, comment out the following line
        predictions = predictions[:,1:,:,:]
        predictions = F.sigmoid(predictions) 
        
        
        # Calculate intersection and union
        intersection = (predictions * gts_one_hot).sum(dim=(2, 3))
        union = (predictions + gts_one_hot).sum(dim=(2, 3)) - intersection
        
        # Calculate mIOU
        miou = (intersection + 1e-6) / (union + 1e-6)
        
        self.miou_scores.append(miou.mean().item())

    def evaluate(self):
        return torch.mean(torch.tensor(self.miou_scores)).item()

class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap
