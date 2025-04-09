# %%
import os
import random
import csv

labeled_samples = 275

# %%
# Define the data directory paths
data_dir = '~./Dataset/0_data_dataset_voc_950/'
IMG_folder_path = data_dir + 'JPEGImages/'
msk_folder_path = data_dir + 'SegmentationClassPNG/'
output_dir = f'~./SSL-CL-RUS/dataset/splits/kidney/'

# %%
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text 

# %%
# Get a list of image file paths and shuffle them
files = [os.path.join(IMG_folder_path, file) for file in os.listdir(IMG_folder_path)]
random.shuffle(files)
files = [remove_prefix(file_path, data_dir) for file_path in files]

# Specify the validation ratio
test_ratio = 0.1
# Define the number of labeled samples

# Calculate the number of samples for the validation set
total_samples = len(files)
test_samples = int(total_samples * test_ratio)

# Split the files into validation, labeled, and unlabeled sets
test_img = files[:test_samples]
labeled_img = files[test_samples:test_samples + labeled_samples]
unlabeled_img = files[test_samples + labeled_samples:]

# %%
print(len(test_img))
print(len(labeled_img))
print(len(unlabeled_img))

# %%
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
import csv
import os

# Assuming the variables `labeled_img`, `val_img`, and `unlabeled_img` are already defined as lists of file paths

# Create corresponding mask file paths for the labeled, validation, and unlabeled images
labeled_mask = [os.path.join(msk_folder_path, os.path.splitext(os.path.basename(file))[0] + '.png') for file in labeled_img]
test_mask = [os.path.join(msk_folder_path, os.path.splitext(os.path.basename(file))[0] + '.png') for file in test_img]
unlabeled_mask = [os.path.join(msk_folder_path, os.path.splitext(os.path.basename(file))[0] + '.png') for file in unlabeled_img]

labeled_mask = [remove_prefix(file_path, data_dir) for file_path in labeled_mask]
test_mask = [remove_prefix(file_path, data_dir) for file_path in test_mask]
unlabeled_mask = [remove_prefix(file_path, data_dir) for file_path in unlabeled_mask]

# Write the labeled image and mask pairs to 'labeled.txt'
with open(os.path.join(output_dir, "labeled.txt"), 'w') as file:
    writer = csv.writer(file, delimiter=' ')
    writer.writerows(zip(labeled_img, labeled_mask))

# Write the validation image and mask pairs to 'val.txt'
with open(os.path.join(output_dir, "test.txt"), 'w') as file:
    writer = csv.writer(file, delimiter=' ')
    writer.writerows(zip(test_img, test_mask))

# %%


# %%



