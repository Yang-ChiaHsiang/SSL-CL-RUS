export semi_setting='kidney'
export unlabeled_num=500

CUDA_VISIBLE_DEVICES=0 python -W ignore main.py \
  --dataset kidney --data-root data/0_data_dataset_voc_950 \
  --batch-size 64 --backbone resnet18 --model deeplabv3plus \
  --labeled-id-path dataset/splits/$semi_setting/train.txt \
  --unlabeled-id-path dataset/splits/$semi_setting/$unlabeled_num/unlabeled.txt \
  --reliable-id-path outdir/reliable_ids/$semi_setting/$unlabeled_num \
  --pseudo-mask-path outdir/pseudo_masks/$semi_setting/$unlabeled_num \
  --save-path outdir/models/$semi_setting --num-unlabeled $unlabeled_num\
  --plus \
  # --PatchCL --contrastiveWeights 0.2 --patch-size 112