

import argparse, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

from data_prep import build_dataframe, split_dataframe, build_generators

IMG_SIZE = (224, 224)
BATCH    = 32

def main(args):
    df = build_dataframe(args.data_dir)
    _, x_val, _, y_val = split_dataframe(df)
    _, val_gen = build_generators([], [], x_val, y_val,
                                  img_size=IMG_SIZE, batch=BATCH)

    model = load_model(args.weights_path)

    preds = model.predict(val_gen, verbose=0).argmax(axis=1)
    true  = val_gen.classes
    class_names = list(val_gen.class_indices.keys())

    print("\nOverall accuracy:",
          accuracy_score(true, preds) * 100, "%\n")

    print("Classification Report:")
    print(classification_report(true, preds, target_names=class_names))

    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--weights_path", required=True,
                   help="Path to saved .h5 weights file")
    args = p.parse_args()
    main(args)
