from glob import glob
import random
import os
import shutil
from tqdm import tqdm

img_paths = glob("images/*")
random.shuffle(img_paths) # in-place

val_fraction = 0.2

val_count = int(val_fraction * len(img_paths))
val_paths, train_paths = img_paths[:val_count], img_paths[val_count:]

os.makedirs("train", exist_ok=True)
for train_path in tqdm(train_paths):
    train_filename = os.path.basename(train_path)
    shutil.copy(train_path, f"train/{train_filename}")
    
os.makedirs("val", exist_ok=True)
for val_path in tqdm(val_paths):
    val_filename = os.path.basename(val_path)
    shutil.copy(val_path, f"val/{val_filename}")
