import argparse
import os
import multiprocessing
from glob import glob
from shutil import copyfile
from typing import List

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from tutorial.io.path_definition import get_project_dir


def file_transfer(new_train: str, jpg_files: List[str], txt_files: List[str], classname_file: str):

    new_train_dir = f"{root_folder}/training/{new_train}"

    if not os.path.isdir(new_train_dir):
        os.makedirs(new_train_dir)

    for ix, (src_jpg_file, src_txt_file) in enumerate(zip(jpg_files, txt_files)):

        p = np.random.rand()

        if p > 0.9:
            dst_jpg_file = f"{new_train_dir}/image_val_{ix:04d}.jpg"
            dst_txt_file = f"{new_train_dir}/image_val_{ix:04d}.txt"
        else:
            dst_jpg_file = f"{new_train_dir}/image_train_{ix:04d}.jpg"
            dst_txt_file = f"{new_train_dir}/image_train_{ix:04d}.txt"

        copyfile(src_jpg_file, dst_jpg_file)
        copyfile(src_txt_file, dst_txt_file)

    copyfile(classname_file, f"{new_train_dir}/classes.names")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--previous_train', dest="PREVIOUS_TRAIN", type=str)
    parser.add_argument('--new_data', dest='NEW_DATA', type=str)
    parser.add_argument('--new_train', dest="NEW_TRAIN", type=str)
    args = parser.parse_args()

    root_folder = f"{get_project_dir()}/data/train/YOLO"

    if args.PREVIOUS_TRAIN:
        train_jpg_files = glob(f"{root_folder}/training/{args.PREVIOUS_TRAIN}/*.jpg")
        train_txt_files = [f.replace("jpg", "txt") for f in train_jpg_files]
    else:
        train_jpg_files = []
        train_txt_files = []

    raw_txt_files = glob(f"{root_folder}/raw/{args.NEW_DATA}/*.txt")
    raw_txt_files = [f for f in raw_txt_files if f.split("/")[-1] != 'classes.txt']
    raw_jpg_files = [f.replace("txt", "jpg") for f in raw_txt_files if os.path.isfile(f.replace("txt", "jpg"))]

    classname_file = f"{root_folder}/raw/{args.NEW_DATA}/classes.txt"

    jpg_files = train_jpg_files + raw_jpg_files
    txt_files = train_txt_files + raw_txt_files

    file_transfer(args.NEW_TRAIN, jpg_files, txt_files, classname_file)

