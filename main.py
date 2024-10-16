# Import  libraries
# Warnings
import warnings
warnings.filterwarnings('ignore')

# System
import os
import gc
import shutil
import time
import glob

# Main 
import random
import numpy as np
import pandas as pd
import json
import cv2
from tqdm import tqdm
tqdm.pandas()
from scipy.io import loadmat

# Data Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from IPython.display import Image, display, HTML

# Creating directory structure, if not exists already
if(os.path.exists("datasets")==False):
    os.mkdir("datasets")

if(os.path.exists("datasets/images")==False):
    os.mkdir("datasets/images")
if(os.path.exists("datasets/images/train")==False):
    os.mkdir("datasets/images/train")
if(os.path.exists("datasets/images/val")==False):
    os.mkdir("datasets/images/val")

if(os.path.exists("datasets/labels")==False):
    os.mkdir("datasets/labels")
if(os.path.exists("datasets/labels/train")==False):
    os.mkdir("datasets/labels/train")
if(os.path.exists("datasets/labels/val")==False):
    os.mkdir("datasets/labels/val")

if(os.path.exists("csv_files")==False):
    os.mkdir("csv_files")
if(os.path.exists("sample_videos")==False):
    os.mkdir("sample_videos")
if(os.path.exists("output_videos")==False):
    os.mkdir("output_videos")

# Depuration mode configuration
debug = False
if(debug):
    train_fol = "3"
    val_fol = "2"
    track_fol = "2"
    epochs = 3
else:
    train_fol = ""
    val_fol = ""
    track_fol = ""
    epochs = 30

# Generate Annotations
def convertBoxFormat(box):
    (box_x_left, box_y_top, box_w, box_h) = box
    (image_w, image_h) = (640, 480)
    dw = 1./image_w
    dh = 1./image_h
    x = (box_x_left + box_w / 2.0) * dw
    y = (box_y_top + box_h / 2.0) * dh
    w = box_w * dw
    h = box_h * dh
    return (x,y,w,h)

annotation_dir = 'kaggle/input/caltechpedestriandataset/annotations/annotations/*'
classes = ['person']
number_of_truth_boxes = 0

img_id_list = []
label_list = []
split_list = []
num_annot_list = []

# Sets
for sets in tqdm(sorted(glob.glob(annotation_dir))):
    set_id = os.path.basename(sets)
    set_number = int(set_id.replace('set',''))
    split_dataset = 'train' if set_number <= 5 else 'val'

    # Videos
    for vid_annotations in sorted(glob.glob(sets + '/*.vbb')):
        video_id = os.path.splitext(os.path.basename(vid_annotations))[0] # Video ID
        vbb = loadmat(vid_annotations) # Read VBB File
        obj_lists = vbb['A'][0][0][1][0] # Annotation list
        obj_lbl = [str(v[0]) for v in vbb['A'][0][0][4][0]] # Label list
        
        # Frames
        for frame_id, obj in enumerate(obj_lists):
            if(len(obj) > 0):
                # Labels
                labels = ''
                num_annot = 0

                for pedestrian_id, pedestrian_pos in zip(obj['id'][0], obj['pos'][0]):
                    pedestrian_id = int(pedestrian_id[0][0]) - 1 # Pedestrian id
                    pedestrian_pos = pedestrian_pos[0].tolist() # Pedestrian BBox

                    # class filter and height filter: here example medium distance
                    if obj_lbl[pedestrian_id] in classes and pedestrian_pos[3] >= 75 and pedestrian_pos[3] <= 250:
                        yolo_box_format = convertBoxFormat(pedestrian_pos) # Convert BBox to YOLO format
                        labels += '0 ' + ' '.join([str(n) for n in yolo_box_format]) + '\n'
                        num_annot += 1
                        number_of_truth_boxes += 1
                
                # Check labels 
                if not labels:
                    continue

                image_id = set_id + '_' + video_id + '_' + f"{frame_id:04d}"
                img_id_list.append(image_id)
                label_list.append(labels)
                split_list.append(split_dataset)
                num_annot_list.append(num_annot)

print('Number of Ground Truth Annotation Box:', number_of_truth_boxes)


df_caltech_annot = pd.DataFrame({
    'image_id': img_id_list,
    'label': label_list,
    'split': split_list,
    'num_annot': num_annot_list
})

df_caltech_annot['set_id'] = df_caltech_annot['image_id'].apply(lambda x: x.split('_')[0])
df_caltech_annot['video_id'] = df_caltech_annot['image_id'].apply(lambda x: x.split('_')[1])
df_caltech_annot['frame_id'] = df_caltech_annot['image_id'].apply(lambda x: x.split('_')[2])

df_caltech_annot.to_csv('csv_files/frame_metadata.csv', index = False)
df_caltech_annot

# display(df_caltech_annot.head())

# Filter Image Files
df_set_video = df_caltech_annot.groupby(['set_id', 'video_id', 'split'])['image_id'].count().reset_index()
df_set_video = df_set_video.rename(columns={'image_id':'total_image'})

df_set_video_train = df_set_video[df_set_video['split']=='train'].reset_index(drop=True)
df_set_video_val = df_set_video[df_set_video['split']=='val'].reset_index(drop=True)

display(df_set_video_train.head())
display(df_set_video_val.head())

total_train_image = sum(df_set_video_train['total_image'])
total_val_image = sum(df_set_video_val['total_image'])
print('Number of train:', total_train_image)
print('Number of Val:', total_val_image)

df_set_video_train = df_set_video_train.groupby('set_id')['video_id'].count().reset_index()
df_set_video_val = df_set_video_val.groupby('set_id')['video_id'].count().reset_index()
df_set_video_count = pd.concat([df_set_video_train, df_set_video_val]).reset_index(drop=True)
df_set_video_count = df_set_video_count.rename(columns={'video_id':'total_video'})
display(df_set_video_count)

df_train_filtered = pd.DataFrame()
df_val_filtered = pd.DataFrame()
set_id_list = list(df_caltech_annot['set_id'].unique())
for i, set_id in enumerate(set_id_list):
    df_set_id = df_set_video[df_set_video['set_id']==set_id]
    video_id_list = list(df_set_id['video_id'].unique())

    for j, vid_id in enumerate(video_id_list):
        df_video_id = df_caltech_annot[(df_caltech_annot['set_id']==set_id)&(df_caltech_annot['video_id']==vid_id)].reset_index(drop=True)
        frame_total = df_video_id.shape[0]
        if(i<=5): # 1000 train images
            limit = int(round((frame_total/total_train_image)*1000, 0))
            df_video_id = df_video_id[:limit]
            df_train_filtered = pd.concat([df_train_filtered,df_video_id], ignore_index=True)
        else:
            limit = int(round((frame_total/total_val_image)*2500, 0))
            df_video_id = df_video_id[:limit]
            df_val_filtered = pd.concat([df_val_filtered,df_video_id], ignore_index=True)

df_train_filtered = df_train_filtered.reset_index(drop=True)
df_val_filtered = df_val_filtered.reset_index(drop=True)

display(df_train_filtered)
display(df_val_filtered)

df_train_filtered.to_csv('csv_files/train_frame_filtered.csv', index=False)
df_val_filtered.to_csv('csv_files/val_frame_filtered.csv', index=False)

# 2.3 Generate Image Files
# https://github.com/simonzachau/caltech-pedestrian-dataset-to-yolo-format-converter
# Generate images from video files
def save_img(dir_path, fn, i, frame):
    cv2.imwrite('{}/{}_{}_{}.png'.format(dir_path, \
        os.path.basename(dir_path), \
        os.path.basename(fn).split('.')[0], \
        f'{i:04d}'), frame)

def convert_caltech(split, df):
    # Directory path
    print(split)
    input_dir = '/kaggle/input/caltechpedestriandataset'
    output_dir = 'datasets/images'
    output_dir = os.path.join(output_dir, 'train') if split=='Train' else os.path.join(output_dir, 'val')
    output_dir = os.path.join(output_dir, 'caltechpedestriandataset')
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    # Sets
    sets_list = sorted(glob.glob(os.path.join(input_dir, split+'/*')))
    print('Total sets:', len(sets_list))
    for dname in sets_list:
        print(dname)
        dname2 = dname.split('/')[-1]
        output_dir2 = os.path.join(output_dir, dname2)
        if(not os.path.exists(output_dir2)):
            os.mkdir(output_dir2)