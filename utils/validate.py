import torch
from tqdm import tqdm
from utils.utils import MeanIOU, DiceCoefficient

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation(model, val_loader, criterion):
    model.eval()
    model.module.contrast = False
    val_loss = 0.0

    dice_coeff = DiceCoefficient()
    miou_metric = MeanIOU()

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc='Validating', leave=True):

            imgs = imgs.to(dev)
            masks = masks.to(dev)

            out = model(imgs)

            loss = criterion(out, masks)
            dice_coeff.add_batch(out, masks)
            miou_metric.add_batch(out, masks)
            
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    
    val_dice = dice_coeff.evaluate()
    val_miou = miou_metric.evaluate()

    return val_loss, val_miou, val_dice
