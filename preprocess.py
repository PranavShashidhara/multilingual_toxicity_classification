import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# -----------------------------
# Function: Preprocess Jigsaw English Dataset
# -----------------------------
def preprocess_jigsaw(processed_folder="data/processed/jigsaw", val_split=0.1, random_state=42):
    os.makedirs(processed_folder, exist_ok=True)
    
    print("Loading English Jigsaw dataset...")
    jigsaw = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge")
    
    train_df = jigsaw["train"].to_pandas()
    test_df = jigsaw["test"].to_pandas()
    
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    any_ones = train_df[label_cols].any()
    print("Any 1s in each label column:\n", any_ones)
    
    label_counts = train_df[label_cols].sum()
    print("\nNumber of positive examples in each label column:\n", label_counts)
    
    # Minimal text cleaning
    for df in [train_df, test_df]:
        df["comment_text"] = df["comment_text"].astype(str).str.strip().str.replace("\n", " ")
    
    # Train/validation split
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_split,
        random_state=random_state,
        stratify=train_df[label_cols].apply(lambda x: x.any(), axis=1)
    )
    
    # Save CSVs
    train_csv = os.path.join(processed_folder, "jigsaw_train.csv")
    val_csv = os.path.join(processed_folder, "jigsaw_val.csv")
    test_csv = os.path.join(processed_folder, "jigsaw_test.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\nSaved Jigsaw preprocessed data to {processed_folder}")
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


# -----------------------------
# Function: Preprocess Multilingual Toxic Dataset (non-English merged)
# -----------------------------
def preprocess_multilingual(processed_folder="data/processed/multilingual_toxic", test_size=0.2, random_state=42):
    os.makedirs(processed_folder, exist_ok=True)
    
    print("Loading TextDetox multilingual toxicity dataset...")
    multilingual_dataset = load_dataset("textdetox/multilingual_toxicity_dataset")
    
    all_non_en = []
    
    # Process non-English languages
    for lang, ds in multilingual_dataset.items():
        if lang == "en":
            continue
        df = ds.to_pandas()
        df["text"] = df["text"].astype(str).str.strip().str.replace("\n", " ")
        df["language"] = lang
        all_non_en.append(df)
    
    # Merge all non-English
    merged_df = pd.concat(all_non_en, ignore_index=True)
    
    # 80/20 train/test split
    if merged_df["toxic"].nunique() > 1:
        train_df, test_df = train_test_split(
            merged_df,
            test_size=test_size,
            stratify=merged_df["toxic"],
            random_state=random_state
        )
    else:
        train_df = merged_df
        test_df = pd.DataFrame(columns=merged_df.columns)
    
    # Save merged CSVs
    train_path = os.path.join(processed_folder, "merged_non_en_train.csv")
    test_path = os.path.join(processed_folder, "merged_non_en_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Merged non-English multilingual dataset saved:")
    print(f"Train: {len(train_df)} rows → {train_path}")
    print(f"Test: {len(test_df)} rows → {test_path}")
    
    # Optional: save per-language test sets
    per_lang_test_folder = os.path.join(processed_folder, "per_language_test")
    os.makedirs(per_lang_test_folder, exist_ok=True)
    
    for lang, df in zip([l for l in multilingual_dataset.keys() if l != "en"], all_non_en):
        if df["toxic"].nunique() > 1:
            _, lang_test_df = train_test_split(
                df,
                test_size=test_size,
                stratify=df["toxic"],
                random_state=random_state
            )
        else:
            lang_test_df = pd.DataFrame(columns=df.columns)
        
        lang_test_path = os.path.join(per_lang_test_folder, f"{lang}_test.csv")
        lang_test_df.to_csv(lang_test_path, index=False)
    
    return train_df, test_df

# -----------------------------
# Main function to call both
# -----------------------------
def main():
    print("===== Preprocessing Jigsaw English Dataset =====")
    preprocess_jigsaw()
    
    print("\n===== Preprocessing Multilingual Toxic Dataset =====")
    preprocess_multilingual()


if __name__ == "__main__":
    main()