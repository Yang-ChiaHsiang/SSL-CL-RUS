# %%
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # %% [markdown]
    # # 讀取 VOC 中有 labeled 的數據

    # %%
    image_dir = "../../Dataset/0_data_dataset_voc_950/JPEGImages"

    # 取得所有檔案名稱，去除副檔名
    file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    ]

    # 轉成 DataFrame
    df = pd.DataFrame(file_names, columns=["filename"])

    # 顯示前幾筆資料
    print(df.head())

    # %% [markdown]
    # # 讀取包含 CKD_Stage 得數據

    # %%
    # 讀取原始資料
    CKD_df = pd.read_csv('../row_data.csv')

    # 依 ID 去除重複，只保留第一筆
    unique_df = CKD_df[['ID', 'c_eGFR']].dropna(subset=['ID']).drop_duplicates(subset='ID')

    # 檢查結果
    print(unique_df.head())
    print(f"Unique IDs with c_eGFR count: {len(unique_df)}")

    # %% [markdown]
    # # 合併兩個 dataframe

    # %%
    unique_df['ID'] = unique_df['ID'].astype(str)  # 確保是字串

    # 提取 prefix 並轉為比對用的 ID 格式
    df['prefix'] = df['filename'].str.extract(r'^(\d+_\d+)')[0].str.replace('_', '', regex=False)

    # 過濾 df 中 prefix 有對應到 unique_df['ID'] 的項目
    filtered_df = df[df['prefix'].isin(unique_df['ID'])]

    # 合併 c_eGFR（left join 保留 filtered_df 中的資料）
    merged_df = pd.merge(filtered_df, unique_df, how='left', left_on='prefix', right_on='ID')

    # 刪除多餘欄位（如 prefix 和合併後重複的 ID 欄）
    merged_df = merged_df.drop(columns=['prefix', 'ID'])

    # 顯示結果
    print(merged_df.head())
    print(f"Merged DataFrame count: {len(merged_df)}")

    # %%
    def classify_ckd_stage(egfr):
        if egfr >= 90:
            return 'G1'
        elif egfr >= 60:
            return 'G2'
        elif egfr >= 45:
            return 'G3a'
        elif egfr >= 30:
            return 'G3b'
        elif egfr >= 15:
            return 'G4'
        else:
            return 'G5'

    # 假設你要對 merged_df 加上 CKD Stage（如果還在 unique_df，也一樣處理即可）
    merged_df['CKD_Stage'] = merged_df['c_eGFR'].apply(classify_ckd_stage)

    # 顯示前幾筆
    print(merged_df[['filename', 'c_eGFR', 'CKD_Stage']].head())
    print(f"Final DataFrame count: {len(merged_df)}")
    merged_df.to_csv('../labeled_data_ckd_stage.csv', index=False)


    # 使用 merged_df（已包含 CKD_Stage）
    df = merged_df.copy()

    # 檢查 CKD_Stage 是否有缺失
    df = df.dropna(subset=['CKD_Stage'])

    print(f"Total data count: {len(df)}")
    print("CKD Stage distribution in full dataset:")
    print(df['CKD_Stage'].value_counts(normalize=True).round(3))

    # 隨機切分: 285 train, 190 val+test（不考慮 CKD_Stage 分布）
    train_df, temp_df = train_test_split(df, train_size=285, test_size=190, random_state=42, shuffle=True)

    # 再從 temp_df 隨機分出 95 val 和 95 test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

    # 重設索引
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # 顯示數量
    print(f"\nDataset split results:")
    print(f"Training: {len(train_df)}, Validation: {len(val_df)}, Testing: {len(test_df)}")

    # 顯示各階段分布（參考用）
    print("\nCKD Stage distribution after random split:")
    print("Train CKD Stage:")
    print(train_df['CKD_Stage'].value_counts(normalize=True).round(3))
    print("\nVal CKD Stage:")
    print(val_df['CKD_Stage'].value_counts(normalize=True).round(3))
    print("\nTest CKD Stage:")
    print(test_df['CKD_Stage'].value_counts(normalize=True).round(3))

    # %%
    def export_split(df, out_path):
        df_out = pd.DataFrame()
        df_out['image_path'] = 'JPEGImages/' + df['filename'] + '.jpg'
        df_out['label_path'] = 'SegmentationClassPNG/' + df['filename'] + '.png'
        df_out.to_csv(out_path, index=False, header=False, sep=' ')

    # 輸出三個 CSV 檔
    export_split(train_df, '../dataset/splits/kidney/train.txt')
    export_split(val_df, '../dataset/splits/kidney/val.txt')
    export_split(test_df, '../dataset/splits/kidney/test.txt')

    print("CSV 檔案已儲存完成 ✅")
# %%
if __name__ == '__main__':
    main()
