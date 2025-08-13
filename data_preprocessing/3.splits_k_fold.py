# %%
import os
import numpy as np

def main():
    dataset = 'kidney'
    dataset_path = f'../dataset/splits/{dataset}'

    original_train_path = f'{dataset_path}/train.txt'
    original_val_path = f'{dataset_path}/val.txt'

    # 讀取檔案
    def read_txt(path):
        with open(path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        return lines

    train_lines = read_txt(original_train_path)  # 285 samples
    val_lines = read_txt(original_val_path)      # 95 samples
    print(f"Train: {len(train_lines)}, Val: {len(val_lines)}")
    print(f"Total: {len(train_lines) + len(val_lines)}, Expecting split: {len(train_lines)} train / {len(val_lines)} val")

    # Combine and shuffle all lines
    all_lines = np.array(train_lines + val_lines)
    rng = np.random.default_rng(42)
    shuffled_indices = rng.permutation(len(all_lines))

    # Set k_fold to 5 and ensure exact 285:95 ratio per fold
    k_fold = 5
    fold_size = len(val_lines)  # Size of validation set per fold (95 samples)

    for fold in range(k_fold):
        # Calculate the start and end indices for validation
        start_val = fold * fold_size
        end_val = start_val + fold_size

        # Ensure end_val does not exceed the total size of the data
        if end_val > len(all_lines):
            end_val = len(all_lines)
            start_val = end_val - fold_size

        val_idx = shuffled_indices[start_val:end_val]
        train_idx = np.concatenate([shuffled_indices[:start_val], shuffled_indices[end_val:]])

        # Ensure the train set has exactly 285 samples
        train_idx = train_idx[:285]

        fold_train_lines = all_lines[train_idx]
        fold_val_lines = all_lines[val_idx]

        fold_dir = os.path.join(dataset_path, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        # Save the training and validation files for this fold
        with open(os.path.join(fold_dir, 'train.txt'), 'w') as f_train, \
            open(os.path.join(fold_dir, 'val.txt'), 'w') as f_val:
            for line in fold_train_lines:
                f_train.write(line + '\n')
            for line in fold_val_lines:
                f_val.write(line + '\n')

        print(f"[Fold {fold}] Train: {len(fold_train_lines)}, Val: {len(fold_val_lines)}")


# %%



if __name__ == '__main__':
    main()