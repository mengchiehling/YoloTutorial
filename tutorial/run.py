import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


from tutorial.yolo_wrapper import YoloInferenceWrapper
from tutorial.io.path_definition import get_project_dir


def plot_image(f: str, box: np.ndarray, label: str, label_proba: float):

    img = Image.open(f)

    x, y, width, height = box
    rect = patches.Rectangle((x, y), width, height, linewidth=6, edgecolor='r', facecolor='none')

    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img)

    label_proba = np.round(label_proba/100, 3)

    ax.add_patch(rect)
    ax.annotate(f"{label}, {label_proba}", xy=(x+width*0.5, 0.9 * y), fontsize=36, c='g')

    fig.patch.set_visible(False)
    ax.axis('off')

    plt.savefig(os.path.join(get_project_dir(), 'data', 'evaluation', 'yolov3-demo.jpg'), dpi=300)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_code', dest="CFG_CODE", type=str)
    parser.add_argument('--model_code', dest="MODEL_CODE", type=str)
    parser.add_argument('--filename', dest='FILENAME', type=str)
    args = parser.parse_args()

    detector = YoloInferenceWrapper(cfg_code=args.CFG_CODE, model_code=args.MODEL_CODE)

    full_file_path = os.path.join(get_project_dir(), 'data', 'images', args.FILENAME)

    label, label_proba, box = detector.predict(img_file=full_file_path)
    plot_image(full_file_path, box, label, label_proba)