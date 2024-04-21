### Python script to split a labeled image dataset into Train, Validation, and Test folders.
# Author: Nafiu Yakubu [Software Engineer, Full Stack Developer, Data Analyst, Database Engineer, AI/ML, DevOps, IaC,]
# Date: 4/21/2024
# Randomly splits images to 80% train, 10% validation, and 10% test, and moves them to their respective folders. 

from pathlib import Path
import random
import os
import sys
import argparse


parser = argparse.ArgumentParser(description="Python Image Data Set Splitter")
parser.add_argument("--image_path", help="All Image directory", default="/dataset/images/all")
parser.add_argument("--train_path", help="Train directory", default="/dataset/images/train")
parser.add_argument("--val_path", help="Validation directory", default="/dataset/images/validation")
parser.add_argument("--test_path", help="Test directory",default="/dataset/images/test")
parser.add_argument("--train_percent", help="Extracted directory", default=0.8)
parser.add_argument("--val_percent", help="Extracted directory", default=0.1 )
parser.add_argument("--test_percent", help="Extracted directory", default=0.1)
args = parser.parse_args()

# Define paths to image folders
image_path = args.image_path #= '/dataset/images/all'
train_path  = args.train_path #= '/dataset/images/train'
val_path  = args.val_path #= '/dataset/images/validation'
test_path = args.test_path # = '/dataset/images/test'

# Define Percentage of file to move to each folders
train_percent_val = args.train_percent  # By Default 80% of the files go to train
val_percent_val = args.val_percent # By Default 10% go to validation
test_percent_val = args.test_percent # By Default 10% go to test


def split_images(image_path, train_path, val_path, test_path, train_percent_val, val_percent_val, test_percent_val):
    # Get list of all images
    jpeg_file_list = [path for path in Path(image_path).rglob('*.jpeg')]
    jpg_file_list = [path for path in Path(image_path).rglob('*.jpg')]
    png_file_list = [path for path in Path(image_path).rglob('*.png')]
    bmp_file_list = [path for path in Path(image_path).rglob('*.bmp')]

    if sys.platform == 'linux':
        JPEG_file_list = [path for path in Path(image_path).rglob('*.JPEG')]
        JPG_file_list = [path for path in Path(image_path).rglob('*.JPG')]
        file_list = jpg_file_list + JPG_file_list + png_file_list + bmp_file_list + JPEG_file_list + jpeg_file_list
    else:
        file_list = jpg_file_list + png_file_list + bmp_file_list + jpeg_file_list

    file_num = len(file_list)
    print('Total images: %d' % file_num)

    # Determine number of files to move to each folder
    train_percent = train_percent_val
    val_percent = val_percent_val 
    test_percent = test_percent_val
    train_num = int(file_num * train_percent)
    val_num = int(file_num * val_percent)
    test_num = file_num - train_num - val_num
    print('Images moving to train: %d' % train_num)
    print('Images moving to validation: %d' % val_num)
    print('Images moving to test: %d' % test_num)

    # Select train_percent% of files randomly and move them to train folder
    for i in range(train_num):
        move_me = random.choice(file_list)
        fn = move_me.name
        base_fn = move_me.stem
        parent_path = move_me.parent
        xml_fn = base_fn + '.xml'
        os.rename(move_me, train_path + '/' + fn)
        os.rename(os.path.join(parent_path, xml_fn), os.path.join(train_path, xml_fn))
        file_list.remove(move_me)

    # Select val_percent% of remaining files and move them to validation folder
    for i in range(val_num):
        move_me = random.choice(file_list)
        fn = move_me.name
        base_fn = move_me.stem
        parent_path = move_me.parent
        xml_fn = base_fn + '.xml'
        os.rename(move_me, val_path + '/' + fn)
        os.rename(os.path.join(parent_path, xml_fn), os.path.join(val_path, xml_fn))
        file_list.remove(move_me)

    # Move remaining files to test folder
    for i in range(test_num):
        move_me = random.choice(file_list)
        fn = move_me.name
        base_fn = move_me.stem
        parent_path = move_me.parent
        xml_fn = base_fn + '.xml'
        os.rename(move_me, test_path + '/' + fn)
        os.rename(os.path.join(parent_path, xml_fn), os.path.join(test_path, xml_fn))
        file_list.remove(move_me)

#Run the splitting function
split_images(image_path, train_path, val_path, test_path, train_percent_val, val_percent_val, test_percent_val)