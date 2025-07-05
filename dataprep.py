

import os, argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_dataframe(root):
    labels = os.listdir(root)
    rows = []
    for label in labels:
        img_dir = os.path.join(root, label)
        for img in os.listdir(img_dir):
            rows.append({"img_path": os.path.join(img_dir, img), "label": label})
    df = pd.DataFrame(rows).sample(frac=1).reset_index(drop=True)  # shuffle
    return df

def split_dataframe(df, test_size=0.2, seed=42):
    x_train, x_val, y_train, y_val = train_test_split(
        df["img_path"], df["label"], test_size=test_size, random_state=seed, stratify=df["label"]
    )
    return x_train.tolist(), x_val.tolist(), y_train.tolist(), y_val.tolist()

def build_generators(x_train, y_train, x_val, y_val, img_size=(224,224), batch=32):
    train_df = pd.DataFrame({"img_path": x_train, "label": y_train})
    val_df   = pd.DataFrame({"img_path": x_val,   "label": y_val})

    datagen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen_val = ImageDataGenerator(rescale=1./255)

    train_gen = datagen_train.flow_from_dataframe(
        train_df, x_col="img_path", y_col="label",
        target_size=img_size, class_mode="categorical",
        batch_size=batch, shuffle=True, seed=42
    )
    val_gen = datagen_val.flow_from_dataframe(
        val_df, x_col="img_path", y_col="label",
        target_size=img_size, class_mode="categorical",
        batch_size=batch, shuffle=False
    )
    return train_gen, val_gen

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Root folder containing class sub‑directories")
    args = p.parse_args()

    df = build_dataframe(args.data_dir)
    print(df.head(), "\n")
    print(f"Total images: {len(df)} | Classes: {df['label'].nunique()}")
