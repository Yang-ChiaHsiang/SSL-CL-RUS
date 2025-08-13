# SSL-CL-RUS

![image](https://github.com/Yang-ChiaHsiang/SSL-CL-RUS/blob/main/SSL-CL-RUS%20Architecture.png)

## Project Description

This project focuses on kidney ultrasound image segmentation using Patch-Level Contrastive Learning (PLCL) and the ST++ method to enhance the learning process. The PLCL Loss function is implemented based on the approach described in the [PatchCL-MedSeg repository](https://github.com/hritam-98/PatchCL-MedSeg). The ST++ method further improves semi-supervised learning by leveraging unlabeled data effectively.

## Features

- **Patch-Level Contrastive Learning**: Improves feature representation for medical image segmentation tasks.
- **Kidney Ultrasound Segmentation**: Specifically tailored for kidney ultrasound images.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Yang-ChiaHsiang/SSL-CL-RUS.git
   cd SSL-CL-RUS
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

The segmentation label dataset follows the PASCAL VOC Dataset format. Below are the steps to preprocess the data:

1. **Dataset Splitting**: Use the script `./data_preprocessing/1_dataset_split.py` to split the dataset into `labeled.txt` and `test.txt`.

2. **Create Unlabeled Dataset**: Run the script `2_create_unlabeled_data.py` to generate the unlabeled dataset.

3. **Data Preprocessing**: Execute `3_data_preprocessing.py` to further split the data into `train.txt` and `val.txt`. Example usage:

   ```bash
   python data_preprocessing.py \
   --dataset_path ~/Dataset/0_data_dataset_voc_950 \
   --voc_output_dir dataset/splits/kidney \
   --voc_splits 500 \
   --crop_output_dir data/0_data_dataset_voc_950 \
   --img_size 448
   ```

### Training

To train the model using the Patch-Level Contrastive Learning (PLCL) approach and the ST++ method, follow the steps below:

1. Set the number of unlabeled samples and dataset name as environment variables:

   ```bash
   export semi_setting='kidney'
   export unlabeled_num=900
   ```

2. Run the training script with the specified parameters:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
   --dataset kidney --data-root data/0_data_dataset_voc_950 \
   --batch-size 64 --backbone resnet18 --model deeplabv3plus \
   --labeled-id-path dataset/splits/$semi_setting/train.txt \
   --unlabeled-id-path dataset/splits/$semi_setting/$unlabeled_num/unlabeled.txt \
   --reliable-id-path outdir/reliable_ids/$semi_setting/$unlabeled_num \
   --pseudo-mask-path outdir/pseudo_masks/$semi_setting/$unlabeled_num \
   --save-path outdir/models/$semi_setting --num-unlabeled $unlabeled_num\
   --plus \
   --PatchCL --contrastiveWeights 0.2 --patch-size 112
   ```

This script trains the model using a semi-supervised learning approach, leveraging both labeled and unlabeled data. The `PatchCL` flag enables Patch-Level Contrastive Learning, which enhances feature representation for segmentation tasks.

## Acknowledgments

This project is inspired by the work in the [PatchCL-MedSeg repository](https://github.com/hritam-98/PatchCL-MedSeg). Special thanks to the contributors of that project for their valuable insights.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this codebase or any part of it in your research, please kindly cite the following paper:

```
@inproceedings{yang2025sslclruskidney,
   title={SSL-CL-RUS: A Semi-Supervised Framework for Renal Ultrasound Segmentation in CKD: Combining Pseudo-Label Guided Contrastive Learning with ST++},
   author={Chia-Hsiang Yang and Wei-Cheng Tseng and Yi-Chin Chen and Jing-Ru He and Kung-Hao Liang and Yen-Hua Huang},
   booktitle={Proceedings of EMBC 2025},
   year={2025}
}
```

We appreciate your support and contributions!
