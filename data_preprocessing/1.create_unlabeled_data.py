# %%
import os
import random
from shutil import copyfile

def main()
    # %%
    os.chdir(os.path.expanduser("~/PLGCL-ST-"))
    print("Current working directory:", os.getcwd())

    # %%
    number_of_unlabeled = 700
    unlabeled_txt_output_path = f'dataset/splits/kidney/{number_of_unlabeled}/unlabeled.txt'
    unlabeled_dataset_path = os.path.expanduser('~/Dataset/0_data_dataset_voc_950/SIEMENS_US_kidney')

    # %%
    output_dir = os.path.dirname(unlabeled_txt_output_path)
    os.makedirs(output_dir, exist_ok=True)

    # %%
    files = [file for file in os.listdir(unlabeled_dataset_path) if file.endswith('.jpg')]

    print(len(files))

    # %%
    # Get a list of JPG files from the dataset
    files = [file for file in os.listdir(unlabeled_dataset_path) if file.endswith('.jpg')]
    # Shuffle and select the specified number of files
    random.shuffle(files)
    selected_files = files[:number_of_unlabeled]

    # %%
    # Process selected files: copy and rename JPG to PNG
    for file in selected_files:
        jpg_path = os.path.join(unlabeled_dataset_path, file)
        png_path = os.path.join(unlabeled_dataset_path, file.replace('.jpg', '.png'))
        
        # Copy the file and rename to .png
        copyfile(jpg_path, png_path)


    # %%
    # Prepare the content for the output file
    output_lines = [
        f"SIEMENS_US_kidney/{file} SIEMENS_US_kidney/{file.replace('.jpg', '.png')}" 
        for file in selected_files
    ]

    # %%
    # Write the selected file paths to the output file
    with open(unlabeled_txt_output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Unlabeled data paths written to {unlabeled_txt_output_path}")

    # %%

if __name__ == '__main__':
    main()
