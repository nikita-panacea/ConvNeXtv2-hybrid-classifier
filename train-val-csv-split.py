from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(csv_path, out_dir, seed=42):
    df = pd.read_csv(csv_path)

    if "label" in df.columns:
        y = df["label"]
    else:
        y = df[ISIC2019Dataset.CLASS_NAMES].values.argmax(axis=1)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(df, y))

    temp_df = df.iloc[temp_idx]
    y_temp = y[temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_df, y_temp))

    df.iloc[train_idx].to_csv(f"{out_dir}/train.csv", index=False)
    temp_df.iloc[val_idx].to_csv(f"{out_dir}/val.csv", index=False)
    temp_df.iloc[test_idx].to_csv(f"{out_dir}/test.csv", index=False)
