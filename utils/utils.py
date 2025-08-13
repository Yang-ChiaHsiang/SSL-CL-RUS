import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_score(pred_logits, target, num_classes=2, ignore_index=255, epsilon=1e-6):
    """
    pred_logits: [B, C, H, W]
    target: [B, H, W]
    """
    B, C, H, W = pred_logits.shape

    # Create a valid mask
    valid_mask = (target != ignore_index)  # [B, H, W]

    # Clone and replace ignored targets with class 0
    target = target.clone()
    target[~valid_mask] = 0

    # One-hot encode target
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
    pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]

    # Expand mask to [B, C, H, W]
    valid_mask = valid_mask.unsqueeze(1).expand(-1, num_classes, -1, -1)

    # Apply mask
    pred_probs = pred_probs * valid_mask
    target_onehot = target_onehot * valid_mask

    # Compute Dice
    dims = (0, 2, 3)
    intersection = torch.sum(pred_probs * target_onehot, dims)
    union = torch.sum(pred_probs + target_onehot, dims)

    dice = (2. * intersection + epsilon) / (union + epsilon)  # [C]

    # Only foreground classes (exclude background: class 0)
    if num_classes > 1:
        return dice[1:].mean()
    else:
        return dice.mean()  # binary segmentation fallback


class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, ignore_index=255, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, pred_logits, target):
        dice = dice_score(pred_logits, target,
                          num_classes=self.num_classes,
                          ignore_index=self.ignore_index,
                          epsilon=self.epsilon)
        return 1 - dice

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


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
    if dataset == 'kidney':
        cmap[0] = np.array([0, 0, 0])       # 背景
        cmap[1] = np.array([128, 0, 0])

    elif dataset == 'pascal' or dataset == 'coco':
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
        cmap[0] = np.array([128, 0, 0])
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
