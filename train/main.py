import argparse
from copy import deepcopy
import numpy as np
import os
import json
from PIL import Image
import torch
from torch.nn import  DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import warnings
from torch.utils.tensorboard import SummaryWriter
import sys
import datetime

project_path = os.path.expanduser("~/ST-PlusPlus-Louis")
sys.path.append(project_path)
os.chdir(project_path)
print("Current working directory:", os.getcwd())

warnings.filterwarnings("ignore")

from dataset.semi_cv2 import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from utils.utils import MeanIOU, meanIOU, DiceCoefficient, calculate_metrics_and_confusion_matrix
from utils.DICELOSS import DiceLoss
from utils.loss_file import save_loss
from utils.validate import validation
from utils.set_seed import seed_torch, seed_worker

# patchCL
from utils.patchCL.queues import Embedding_Queues
from utils.patchCL.patch_utils import _get_patches
from utils.patchCL.aug_utils import batch_augment
from utils.patchCL.get_embds import get_embeddings
from utils.patchCL.plg_loss import PCGJCL



MODE = None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getColormap():
    with open('voc_mask_color_map.json', 'r') as file:
        JsonData = json.load(file)
    voc_mask_color_map = JsonData['voc_mask_color_map']
    return voc_mask_color_map

def parse_args():
    color_map = getColormap()
    
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes', 'kidney'], default='kidney')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'resnet101'], default='resnet18')
    parser.add_argument('--num-classes', type=int, default=len(color_map))
    parser.add_argument('--model', type=str, choices=['deeplabv3plus'], default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--validation-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-model-path', type=str, required=True)
    parser.add_argument('--unlabeled_data', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true', help='whether to use ST++')
    
    # for PatchCL
    parser.add_argument('--PatchCL', type=str2bool, default=False, help='Patch Level Contrastive Learning')
    parser.add_argument('--reset_bn', type=str2bool, default=False, help='Patch Level Contrastive Learning reset_bn')
    parser.add_argument('--embedding_size', type=int, default=128, help='Size of the embedding vectors')
    parser.add_argument('--contrastiveWeights', type=float, default=0, help='Weight for PatchCL loss')
    parser.add_argument('--patch_size', type=int, default=112, help='Batch size for contrastive learning')
    parser.add_argument('--dynamic', type=str2bool, default=False, help='Dynamic Contrastive loss weight for contrastive learning')
    parser.add_argument('--pretrain', type=str, help='Pretrained model path')

    args = parser.parse_args()
    args.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return args
    
def init_basic_elems(args):
    head_lr_multiple = 10.0
    model = DeepLabV3Plus(args.backbone, args.num_classes, args.embedding_size)

    optimizer = SGD([{'params': model.encoder.parameters(), 'lr': args.lr},  # 修改這裡
                     {'params': [param for name, param in model.named_parameters()
                                 if 'encoder' not in name],  # 修改這裡
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    teacher_model = copy.deepcopy(model)
    model, teacher_model = model.cuda(), teacher_model.cuda()
    return model, teacher_model, optimizer

def main(args):
    global MODE
    MODE = 'train'
    g = torch.Generator()
    g.manual_seed(42)
    seed_torch(42)
    
    voc_mask_color_map = getColormap()

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = DiceLoss()
    valset = SemiDataset(args.dataset, args.data_root, 'val', args.crop_size, args.validation_id_path, colormap=voc_mask_color_map)    
    valloader = DataLoader(valset, batch_size=len(valset), shuffle=True, pin_memory=True, num_workers=4)

    testset = SemiDataset(args.dataset, args.data_root, 'test', args.crop_size, colormap=voc_mask_color_map)    
    testloader = DataLoader(testset, batch_size=4, shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))
    
    trainset = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, args.labeled_id_path, colormap=voc_mask_color_map)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=16)
    
    model, teacher_model, optimizer = init_basic_elems(args)
    best_model, best_teacher_model = train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step=f'st{"++" if args.plus else ""}_supervised_labeled')

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')
        g = torch.Generator()
        g.manual_seed(42)
        seed_torch(42)

        dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, args.unlabeled_id_path, colormap=voc_mask_color_map)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        label(best_model, dataloader, args, voc_mask_color_map)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, 'semi_train', args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path, colormap=voc_mask_color_map)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=16)

        model, teacher_model, optimizer = init_basic_elems(args)

        train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step='st-semi-supervised')

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')
    g = torch.Generator()
    g.manual_seed(42)
    seed_torch(42)

    dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, args.unlabeled_id_path, colormap=voc_mask_color_map)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable(best_model, best_teacher_model, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')
    g = torch.Generator()
    g.manual_seed(42)
    seed_torch(42)

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, f'mean_teacher_{args.reset_bn}_{args.patch_size}_{args.contrastiveWeights}_reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, cur_unlabeled_id_path, colormap=voc_mask_color_map)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args, voc_mask_color_map)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')
    g = torch.Generator()
    g.manual_seed(42)
    seed_torch(42)

    MODE = 'semi_train'
    trainset = SemiDataset(args.dataset, args.data_root, 'semi_train', args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path, colormap=voc_mask_color_map)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=16)

    model, teacher_model, optimizer = init_basic_elems(args)

    best_model = train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step='re-training-1st')

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')
    g = torch.Generator()
    g.manual_seed(42)
    seed_torch(42)

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, f'mean_teacher_{args.reset_bn}_{args.patch_size}_{args.contrastiveWeights}_unreliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', args.crop_size, None, cur_unlabeled_id_path, colormap=voc_mask_color_map)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args, voc_mask_color_map)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')
    g = torch.Generator()
    g.manual_seed(42)
    seed_torch(42)
    trainset = SemiDataset(args.dataset, args.data_root, 'semi_train', args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path, colormap=voc_mask_color_map)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=16)

    model, teacher_model, optimizer = init_basic_elems(args)

    train(model, teacher_model, trainloader, valloader, valloader, criterion, optimizer, args, step='re-training-2st')



def reset_bn_stats(model, train_loader):
    model.train()
    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.cuda() 
            _ = model(imgs)

def train(model, teacher_model, trainloader, valloader, testloader, criterion, optimizer, args, step=""):
    # log file
    if args.PatchCL:
        log_file_name = f'Unlabeled({args.unlabeled_data})_simple_PLGCL-ImageSize({args.crop_size})_PatchSize({args.patch_size})_CLWeight({args.contrastiveWeights})_IsPlus({args.plus})_Step({step})'
    else:
        log_file_name = f'Unlabeled({args.unlabeled_data})ImageSize({args.crop_size})_IsPlus({args.plus})_Step({step})'
    log_file_name = f'{log_file_name}_{args.timestamp}'

    writer_train = SummaryWriter(f'runs/{log_file_name}/train')
    writer_val = SummaryWriter(f'runs/{log_file_name}/val')
    writer_test = SummaryWriter(f'runs/{log_file_name}/test')

    iters = 0
    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0
    global MODE

    embd_queues = Embedding_Queues(args.num_classes)
    for param in teacher_model.parameters():
        param.requires_grad=False
            
    for epoch in range(args.epochs):
        # 動態 Contrastive weight
        train_CL_weight = (epoch / args.epochs) if args.dynamic else args.contrastiveWeights
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()    
        teacher_model.train()
        
        epoch_t_loss = 0.0
        epoch_t_SegLoss = 0.0 
        epoch_t_PatchCLLoss = 0.0
        epoch_t_sensitivity, epoch_t_specificity = 0.0, 0.0
        
        metric_t_miou = MeanIOU()
        metric_t_dice = DiceCoefficient()
        tbar = tqdm(trainloader)
        for i, (img, mask) in enumerate(tbar):
            if args.PatchCL:
                # patchCL
                img, mask = img.cpu(), mask.cpu()
                patch_list = _get_patches(
                    img, mask,
                    classes=args.num_classes,
                    background=True,
                    img_size=args.crop_size,
                    patch_size=args.patch_size
                )
                augmented_patch_list = batch_augment(patch_list, args.batch_size)
                aug_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in augmented_patch_list]
                qualified_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in patch_list]

                model.contrast = True
                student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True, args.batch_size)

                teacher_model = teacher_model.train()
                teacher_model.contrast = True
                teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, True, args.batch_size)

                embd_queues.enqueue(teacher_embedding_list)
                PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, args.embedding_size, 0.2 , 4, psi=4096)  
                PCGJCL_loss = PCGJCL_loss.cuda()
            
            model.contrast = False
            # segmentation
            img, mask = img.cuda(), mask.cuda()
            pred = model(img)

            SegLoss = criterion(pred, mask)
            if not args.PatchCL: PCGJCL_loss = torch.tensor(0.0).cuda() 
            loss =  (1-train_CL_weight)*SegLoss + (train_CL_weight * PCGJCL_loss)
            sensitivity, specificity, TP, TN, FP, FN = calculate_metrics_and_confusion_matrix(pred, mask)
            epoch_t_sensitivity += sensitivity
            epoch_t_specificity += specificity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_t_loss += loss.item()
            epoch_t_SegLoss += SegLoss.item()
            epoch_t_PatchCLLoss += PCGJCL_loss.item()
            metric_t_miou.add_batch(pred, mask)
            metric_t_dice.add_batch(pred, mask)

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (epoch_t_loss / (i + 1)))

            for param_stud, param_teach in zip(model.parameters(), teacher_model.parameters()):
                param_teach.data.copy_(0.001 * param_stud.data + 0.999 * param_teach.data)

        writer_train.add_scalar('Loss', epoch_t_loss / len(trainloader), epoch+1)
        writer_train.add_scalar('SegLoss', epoch_t_SegLoss / len(trainloader), epoch+1)
        writer_train.add_scalar('PatchCLLoss', epoch_t_PatchCLLoss / len(trainloader), epoch+1)
        writer_train.add_scalar('mIoU', metric_t_miou.evaluate(), epoch+1)
        writer_train.add_scalar('Dice', metric_t_dice.evaluate(), epoch+1)
        writer_train.add_scalar('Sensitivity', epoch_t_sensitivity / len(trainloader), epoch+1)
        writer_train.add_scalar('Specificity', epoch_t_specificity / len(trainloader), epoch+1)

        if args.reset_bn: reset_bn_stats(model, trainloader)
        writer_val = validation(model, teacher_model, embd_queues, train_CL_weight, valloader, args, criterion, writer_val, epoch, 'Validation')

    torch.cuda.empty_cache()
    writer_test = validation(model, teacher_model, embd_queues, train_CL_weight, testloader, args, criterion, writer_test, epoch, 'Test')

    if args.PatchCL:
        save_model_path_s = f"{args.save_model_path}/mean_teacher/PLGCL/{log_file_name}_s.pth"
        save_model_path_t = f"{args.save_model_path}/mean_teacher/PLGCL/{log_file_name}_t.pth"
    else:
        save_model_path_s = f"{args.save_model_path}/mean_teacher/{log_file_name}_s.pth"
        save_model_path_t = f"{args.save_model_path}/mean_teacher/{log_file_name}_t.pth"

    # 建立資料夾（假設兩個模型在同一資料夾）
    save_dir = os.path.dirname(save_model_path_s)
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)

    # 儲存模型
    torch.save(model.state_dict(), save_model_path_s)
    torch.save(teacher_model.state_dict(), save_model_path_t)

    writer_train.flush()
    writer_val.flush()
    writer_test.flush()

    if MODE == 'train':
        return model, teacher_model

    return model

def validation(model, teacher_model, embd_queues, train_CL_weight, dataloader, args, criterion, writer, epoch, name):        # validation
    model.eval()
    model.contrast = False
    epoch_v_loss = 0.0
    epoch_v_SegLoss = 0.0
    epoch_v_PatchCLLoss = 0.0
    epoch_v_sensitivity, epoch_v_specificity = 0.0, 0.0

    dice_coeff = DiceCoefficient()
    miou_metric = MeanIOU()
    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc=name, leave=True):
            if args.PatchCL:
                # patchCL
                img, mask = imgs.cpu(), masks.cpu()
                patch_list = _get_patches(
                    img, mask,
                    classes=args.num_classes,
                    background=True,
                    img_size=args.crop_size,
                    patch_size=args.patch_size
                )

                augmented_patch_list = batch_augment(patch_list, args.batch_size)
                aug_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in augmented_patch_list]
                qualified_tensor_patch_list = [torch.tensor(patch) if patch is not None else None for patch in patch_list]

                model.contrast = True
                student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True, args.batch_size)

                teacher_model = teacher_model.train()
                teacher_model.contrast = True
                teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, True, args.batch_size)

                PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, args.embedding_size, 0.2 , 4, psi=4096)  
                PCGJCL_loss = PCGJCL_loss.cuda()


            imgs, masks = imgs.cuda(), masks.cuda()
            model.contrast = False
            pred = model(imgs)

            SegLoss = criterion(pred, masks)
            if not args.PatchCL: PCGJCL_loss = torch.tensor(0.0).cuda()
            loss =  (1-train_CL_weight)*SegLoss + (train_CL_weight * PCGJCL_loss)
            sensitivity, specificity, TP, TN, FP, FN = calculate_metrics_and_confusion_matrix(pred, masks)
            epoch_v_sensitivity += sensitivity
            epoch_v_specificity += specificity

            dice_coeff.add_batch(pred, masks)
            miou_metric.add_batch(pred, masks)
            epoch_v_loss += loss.item()
            epoch_v_SegLoss += SegLoss.item()
            epoch_v_PatchCLLoss += PCGJCL_loss.item()

    writer.add_scalar('Loss', epoch_v_loss / len(dataloader), epoch+1)
    writer.add_scalar('SegLoss', epoch_v_SegLoss / len(dataloader), epoch+1)
    writer.add_scalar('PatchCLLoss', epoch_v_PatchCLLoss / len(dataloader), epoch+1)
    writer.add_scalar('mIoU', miou_metric.evaluate(), epoch+1)
    writer.add_scalar('Dice', dice_coeff.evaluate(), epoch+1)
    writer.add_scalar('Sensitivity', epoch_v_sensitivity / len(dataloader), epoch+1)
    writer.add_scalar('Specificity', epoch_v_specificity / len(dataloader), epoch+1)

    return writer

def select_reliable(student_model, teacher_model, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    models = [student_model, teacher_model]

    for model in models:
        model.eval()
        model.contrast = False
        
    tbar = tqdm(dataloader)

    id_to_reliability = []
    
    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                model.contrast = False
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(args.num_classes)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, f'mean_teacher_{args.reset_bn}_{args.patch_size}_{args.contrastiveWeights}_reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, f'mean_teacher_{args.reset_bn}_{args.patch_size}_{args.contrastiveWeights}_unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args, voc_mask_color_map):
    model.eval()
    model.contrast = False
    tbar = tqdm(dataloader)
    cmap = np.array(voc_mask_color_map, dtype=np.uint8)#.flatten()
    with torch.no_grad():
        for img, mask,id in tbar:
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu()

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)
            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].replace('.jpg','.png'))))

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
