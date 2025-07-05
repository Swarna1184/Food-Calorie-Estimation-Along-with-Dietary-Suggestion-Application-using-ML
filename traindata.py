
import argparse, os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_prep import build_dataframe, split_dataframe, build_generators

IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS   = 10

def build_model(base):
    base.trainable = False
    model = Sequential([
        base,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.25), BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.25), BatchNormalization(),
        Dense(80,  activation='softmax')   # ← 80 classes
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_one(backbone_name, train_gen, val_gen, out_dir):
    if backbone_name == "mobilenet":
        base = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(*IMG_SIZE,3))
    elif backbone_name == "inception":
        base = InceptionV3(weights='imagenet', include_top=False,
                           input_shape=(*IMG_SIZE,3))
    else:
        raise ValueError("Backbone must be 'mobilenet' or 'inception'.")

    model = build_model(base)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = ModelCheckpoint(os.path.join(out_dir, f"{backbone_name}_best.h5"),
                           save_best_only=True, monitor='val_accuracy',
                           mode='max', verbose=1)
    early = EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        validation_data=val_gen,
        validation_steps=len(val_gen),
        epochs=EPOCHS,
        callbacks=[ckpt, early],
        verbose=2
    )
    return history

def main(args):
    df = build_dataframe(args.data_dir)
    x_tr, x_val, y_tr, y_val = split_dataframe(df)

    train_gen, val_gen = build_generators(x_tr, y_tr, x_val, y_val,
                                          img_size=IMG_SIZE, batch=BATCH)

    print("Training MobileNetV2…")
    train_one("mobilenet", train_gen, val_gen, "weights")

    print("\nTraining InceptionV3…")
    train_one("inception", train_gen, val_gen, "weights")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Root folder with class‑named sub‑directories")
    args = p.parse_args()
    main(args)
