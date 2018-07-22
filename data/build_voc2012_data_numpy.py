# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import build_data
import tensorflow as tf
from scipy.misc import imread, imresize

from tqdm import tqdm
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           './VOCdevkit/VOC2012/JPEGImages',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './VOCdevkit/VOC2012/SegmentationClassRaw',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    './VOCdevkit/VOC2012/ImageSets/Segmentation',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    './new_npy',
    'Path to save converted SSTable of TensorFlow examples.')


def _convert_dataset(dataset_split):
    """Converts the specified dataset split to numpy format.

    Args:
      dataset_split: The dataset split (e.g., train, test).

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """
    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    img_np = np.empty((num_images, 160, 160, 3), dtype=np.uint8)
    label_np = np.empty((num_images, 160, 160), dtype=np.uint8)

    for i in tqdm(range(num_images)):
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        # image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        # height, width = image_reader.read_image_dims(image_data)
        image_data = imresize(imread(image_filename), (160, 160), 'bilinear')
        height, width = image_data.shape[0:2]
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        # seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        # seg_height, seg_width = label_reader.read_image_dims(seg_data)
        seg_data = imresize(imread(seg_filename), (160, 160), 'nearest')
        seg_height, seg_width = seg_data.shape[0:2]
        if height != seg_height or width != seg_width:
            raise RuntimeError('Shape mismatched between image and label.')
        # print(image_data.shape)
        # print(image_data.dtype)
        # print(seg_data.shape)
        # print(seg_data.dtype)
        # exit(0)
        img_np[i] = image_data.copy()
        label_np[i] = seg_data.copy()
    np.save(FLAGS.output_dir + 'x_' + dataset, np.transpose(img_np, (0, 3, 1, 2)))
    np.save(FLAGS.output_dir + 'y_' + dataset, label_np)


def main(unused_argv):
    dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
    for dataset_split in dataset_splits:
        _convert_dataset(dataset_split)


if __name__ == '__main__':
    tf.app.run()
