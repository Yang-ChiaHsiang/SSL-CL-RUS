from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils.utils import count_params, meanIOU, color_map, DiceLoss, dice_score

import argparse
from copy import deepcopy
import numpy as np
import os
import copy
from PIL import Image
import random
import datetime
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# patchCL
from utils.patchCL.queues import Embedding_Queues
from utils.patchCL.patch_utils import _get_patches
from utils.patchCL.aug_utils import batch_augment
from utils.patchCL.get_embds import get_embeddings
from utils.patchCL.stochastic_approx import StochasticApprox
from utils.patchCL.plg_loss import PCGJCL, simple_PCGJCL

MODE = None
def seed_torch(seed):
    random.seed(seed)  # 設定 Python 標準隨機庫
    np.random.seed(seed)  # 設定 NumPy 隨機數生成器
    torch.manual_seed(seed)  # 設定 PyTorch CPU 隨機數生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 設定 PyTorch CUDA 隨機數生成器
        torch.cuda.manual_seed_all(seed)  # 設定所有 GPU 設備的隨機數生成器
        torch.backends.cudnn.deterministic = True  # 確保 CuDNN 使用固定的算法
        torch.backends.cudnn.benchmark = False  # 關閉 CuDNN 自適應算法選擇    

def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes', 'kidney'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--num-unlabeled', type=int, default=0, help='Number of unlabeled images for the dataset')

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')
    
    # PLGCL
    parser.add_argument('--contrastiveWeights', type=float, default=0)
    parser.add_argument('--PatchCL', dest='PatchCL', default=False, action='store_true', help='Patch Level Contrastive Learning')
    parser.add_argument('--patch-size', type=int, default=112, help='Batch size for contrastive learning')
    
    # k_fold argument
    parser.add_argument('--k_fold', type=int, default=None, help='K-fold cross-validation fold number')

    args = parser.parse_args()
    if args.k_fold is not None:
        args.timestamp = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_fold_{args.k_fold}"
    else:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
    # Make reliable_id_path and pseudo_mask_path unique for each experiment to avoid conflicts
    if args.plus and args.reliable_id_path:
        base_path = args.reliable_id_path
        if args.PatchCL:
            unique_suffix = f"_PLGCL_patch{args.patch_size}_cw{args.contrastiveWeights}_{args.timestamp}"
        else:
            unique_suffix = f"_baseline_{args.timestamp}"
        args.reliable_id_path = f"{base_path}{unique_suffix}"
    
    # Make pseudo_mask_path unique for each experiment to avoid conflicts
    base_pseudo_path = args.pseudo_mask_path
    if args.PatchCL:
        unique_suffix = f"_PLGCL_patch{args.patch_size}_cw{args.contrastiveWeights}_{args.timestamp}"
    else:
        unique_suffix = f"_baseline_{args.timestamp}"
    args.pseudo_mask_path = f"{base_pseudo_path}{unique_suffix}"
    
    return args

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')
    seed_torch(42)
    criterion = DiceLoss()

    valset = SemiDataset(args.dataset, args.data_root, 'val', args.crop_size)
    valloader = DataLoader(valset, batch_size=args.batch_size,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    test = SemiDataset(args.dataset, args.data_root, 'test', args.crop_size)
    testloader = DataLoader(test, batch_size=args.batch_size,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    
    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global MODE
    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=False, persistent_workers=True, prefetch_factor=2)

    model, teacher_model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    best_model, checkpoints = train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step='supervised')

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

        dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

        label(best_model, dataloader, args)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=8, drop_last=False, persistent_workers=True, prefetch_factor=2)

        model, teacher_model, optimizer = init_basic_elems(args)

        train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step='re-training-1st')

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=False)

    model, teacher_model, optimizer = init_basic_elems(args)

    best_model = train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step='re-training-1st')

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=8, drop_last=False, persistent_workers=True, prefetch_factor=2)

    model, teacher_model, optimizer = init_basic_elems(args)

    train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step='re-training-2st')

def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    num_classes = {'pascal': 21, 'cityscapes': 19, 'kidney': 2}[args.dataset]
    print('Model: %s, Backbone: %s, Num classes: %i' % (args.model, args.backbone, num_classes))
    model = model_zoo[args.model](args.backbone, num_classes, embedding_size=128)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    # 分別收集 backbone 和非 backbone 的參數，確保沒有重疊
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = SGD([{'params': backbone_params, 'lr': args.lr},
                     {'params': head_params, 'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()
    teacher_model = copy.deepcopy(model).cuda()
    return model, teacher_model, optimizer

def reset_bn_stats(model, train_loader):
    model.train()
    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.cuda() 
            _ = model(imgs)

def reset_bn_stats_comprehensive(model, train_loader, num_batches=100):
    """更全面的BN統計重置"""
    model.train()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            imgs = imgs.cuda()
            # 同時運行segmentation和contrast模式
            model.module.contrast = False
            _ = model(imgs)
            if hasattr(model.module, 'contrast'):
                model.module.contrast = True
                _ = model(imgs)
    model.eval()

def update_teacher_model(teacher_model, student_model, momentum=0.999):
    """使用EMA更新teacher模型"""
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data = momentum * teacher_param.data + (1 - momentum) * student_param.data

def train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step):
    num_classes = {'pascal': 21, 'cityscapes': 19, 'kidney': 2}[args.dataset]
    iters = 0
    total_iters = len(trainloader) * args.epochs

    if args.PatchCL:
        log_file_name = f'dataset({args.num_unlabeled})_PLGCL_ImageSize({args.crop_size})_PatchSize({args.patch_size})_CLWeight({args.contrastiveWeights})_IsPlus({args.plus})_Step({step})'
    else:
        log_file_name = f'dataset({args.num_unlabeled})_ImageSize({args.crop_size})_IsPlus({args.plus})_Step({step})'
    log_file_name = f'{log_file_name}_{args.timestamp}'
    print(f'Logging to: {log_file_name}')

    writer_train = SummaryWriter(f'runs/{log_file_name}/train')
    writer_val = SummaryWriter(f'runs/{log_file_name}/val')
    writer_test = SummaryWriter(f'runs/{log_file_name}/test')
    previous_best = 0.0

    global MODE
    checkpoints = []

    stochastic_approx = StochasticApprox(num_classes, 0.5, 0.8, patch_size=args.patch_size)
    embd_queues = Embedding_Queues(num_classes, max_length=2048)

    for param in teacher_model.parameters():
        param.requires_grad=False

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        teacher_model.train()
        train_total_loss = 0.0
        train_total_seg_loss = 0.0
        train_total_PLGCL_loss = 0.0
        train_total_dice_score = 0.0
        train_metric = meanIOU(num_classes=num_classes)
        tbar = tqdm(trainloader)
        for i, (img, mask) in enumerate(tbar):
            if args.PatchCL:
                with torch.no_grad():
                    img, mask = img.cpu(), mask.cpu()
                    patch_list = _get_patches(img, mask, classes=num_classes, background=True, img_size=args.crop_size, patch_size=args.patch_size)

                    if step != 'supervised': patch_list = stochastic_approx.update(patch_list)
                    augmented_patch_list = batch_augment(patch_list, args.batch_size)
                    aug_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in augmented_patch_list]
                    qualified_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in patch_list]

                model.module.contrast = True
                student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True, args.batch_size)

                teacher_model.module.contrast = True
                teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, False, args.batch_size)

                embd_queues.enqueue(teacher_embedding_list)
                PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 0.2 , 4, psi=4096)  
                PCGJCL_loss = PCGJCL_loss.cuda()
            
            img, mask = img.cuda(), mask.cuda()
            model.module.contrast = False
            teacher_model.module.contrast = False
            pred = model(img)
            seg_loss = criterion(pred, mask)
            dice_score_value = dice_score(pred, mask)
            train_total_dice_score += dice_score_value.item()
            train_metric.add_batch(torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy())
            mIOU = train_metric.evaluate()[-1]

            if not args.PatchCL: PCGJCL_loss = torch.tensor(0.0).cuda() 
            loss = seg_loss + args.contrastiveWeights * PCGJCL_loss
            train_total_PLGCL_loss += PCGJCL_loss.item()
            train_total_seg_loss += seg_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_total_seg_loss += seg_loss.item()
            train_total_PLGCL_loss += PCGJCL_loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
            
            update_teacher_model(teacher_model, model)
            tbar.set_description('Loss: %.3f | Dice: %.3f | mIOU: %.3f' % (
                train_total_loss / (i + 1),
                train_total_dice_score / (i + 1),
                mIOU
            ))

        writer_train.add_scalar('Loss', train_total_loss / (i + 1), epoch)
        writer_train.add_scalar('SegLoss', train_total_seg_loss / (i + 1), epoch)
        writer_train.add_scalar('PLGCL_Loss', train_total_PLGCL_loss / (i + 1), epoch)
        writer_train.add_scalar('Dice', train_total_dice_score / (i + 1), epoch)
        writer_train.add_scalar('mIOU', mIOU, epoch)
        
        reset_bn_stats_comprehensive(model, trainloader)
        mIOU = validation(model, teacher_model, valloader, criterion, args, writer_val, embd_queues, epoch)
        mIOU *= 100.0

        save_path = os.path.join(args.save_path, log_file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    # test
    mIOU = validation(best_model, teacher_model, testloader, criterion, args, writer_test, embd_queues, epoch)
    print('\nFinal mIOU on test set: %.2f' % (mIOU * 100.0))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model

def validation(model, teacher_model, valloader, criterion, args, writer_val, embd_queues, epoch):
    iters = 0
    val_total_loss = 0.0
    val_total_seg_loss = 0.0
    val_total_PLGCL_loss = 0.0
    val_total_dice_score = 0.0

    num_classes = {'pascal': 21, 'cityscapes': 19, 'kidney': 2}[args.dataset]
    val_metric = meanIOU(num_classes=num_classes)

    model.eval()
    teacher_model.eval()
    tbar = tqdm(valloader)

    with torch.no_grad():
        for i, (img, mask, _) in enumerate(tbar):
            img, mask = img.cpu(), mask.cpu()
            if args.PatchCL:
                patch_list = _get_patches(img, mask, classes=num_classes, background=True, img_size=args.crop_size, patch_size=args.patch_size)
                augmented_patch_list = batch_augment(patch_list, args.batch_size)
                aug_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in augmented_patch_list]
                qualified_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in patch_list]

                model.module.contrast = True
                student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True, args.batch_size)

                teacher_model.module.contrast = True
                teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, True, args.batch_size)

                embd_queues.enqueue(teacher_embedding_list)
                PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 0.2 , 4, psi=4096)  
                PCGJCL_loss = PCGJCL_loss.cuda()
            
            img, mask = img.cuda(), mask.cuda()
            model.module.contrast = False
            teacher_model.module.contrast = False
            pred = model(img)
            seg_loss = criterion(pred, mask)
            dice_score_value = dice_score(pred, mask)
            val_total_dice_score += dice_score_value.item()
            val_metric.add_batch(torch.argmax(pred, dim=1).cpu().numpy(), mask.cpu().numpy())
            mIOU = val_metric.evaluate()[-1]

            if not args.PatchCL: PCGJCL_loss = torch.tensor(0.0).cuda()
            loss = seg_loss + args.contrastiveWeights * PCGJCL_loss
            val_total_PLGCL_loss += PCGJCL_loss.item()
            val_total_seg_loss += seg_loss.item()   
            val_total_loss += loss.item()

            tbar.set_description('Loss: %.3f | Dice: %.3f | mIOU: %.3f' % (
                val_total_loss / (i + 1),
                val_total_dice_score / (i + 1),
                mIOU
            ))
            iters += 1
        
    writer_val.add_scalar('Loss', val_total_loss / (i + 1), epoch)
    writer_val.add_scalar('SegLoss', val_total_seg_loss / (i + 1), epoch)
    writer_val.add_scalar('PLGCL_Loss', val_total_PLGCL_loss / (i + 1), epoch)
    writer_val.add_scalar('Dice', val_total_dice_score / (i + 1), epoch)
    writer_val.add_scalar('mIOU', mIOU, epoch)

    return mIOU


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
        models[i].module.contrast = False

    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            num_classes = {'pascal': 21, 'cityscapes': 19, 'kidney': 2}[args.dataset]
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=num_classes)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()
    model.module.contrast = False
    tbar = tqdm(dataloader)

    num_classes = {'pascal': 21, 'cityscapes': 19, 'kidney': 2}[args.dataset]
    metric = meanIOU(num_classes=num_classes)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu()  # shape: [B, H, W]

            mask_np = mask.squeeze(0).numpy()
            if mask_np.ndim == 3 and mask_np.shape[2] == 3:
                mask_class = np.all(mask_np == [128, 0, 0], axis=-1).astype(np.uint8)
            else:
                mask_class = mask_np  # already class map

            metric.add_batch(pred.numpy(), mask_class[np.newaxis, ...])  # shape: [B, H, W]
            mIOU = metric.evaluate()[-1]

            pred_img = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred_img.putpalette(cmap)
            pred_img.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))



if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240, 'kidney': 100}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'kidney': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721, 'kidney': 224}[args.dataset]

    print()
    print(args)

    main(args)
