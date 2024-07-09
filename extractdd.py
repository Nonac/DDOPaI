import os
import random
import shutil
import abc
import json
import time
import functools
from pathlib import Path
import argparse
import tensorflow as tf
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras.applications.resnet50 as resnet50

from ACE.tf import ace_run


from _PATH import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IF_RESHUFFLE=False
RANDOM_SEED = 0
random.seed(RANDOM_SEED)
TRAIN_PATH = os.path.join(IMAGENET_PATH, "train")
MODEL_DICT={
    "vgg16":vgg16.VGG16,
    'resnet50':resnet50.ResNet50
}

BN_LAYER="conv5_block3_out"


parser = argparse.ArgumentParser(description='Extract Discriminative Data with ACE.')
parser.add_argument('--teacher_model',default="resnet50", type=str, required=True, help='Path to save the random sampled images.')
parser.add_argument('--gpu',default="1", type=str, required=False, help='gpu index')
parser.add_argument('--working_dir',default="tmp/imagenet", type=str, required=False, help='working dir')
parser.add_argument('--labels_path',default="imagenet1k_labels.txt", type=str, required=False, help='imagenet 1k labels')
parser.add_argument('--bottlenecks',default=BN_LAYER, type=str, required=False, help='bottleneck layer')
parser.add_argument('--num_random_exp',default=20, type=int, required=False, help='Number of random experiments used for statistical testing, etc"')
parser.add_argument('--max_imgs',default=40, type=int, required=False, help='Maximum number of images in a discovered concept')
parser.add_argument('--min_imgs',default=40, type=int, required=False, help='Minimum number of images in a discovered concept')
parser.add_argument('--num_parallel_workers',default=0, type=int, required=False, help='Number of parallel jobs.')
parser.add_argument('--random500',default=500, type=int, required=False, help='each one contains 500 randomly selected images from the data set')

config = parser.parse_args()



class imagenet(abc.ABC):
    def __init__(self, config):
        self.PATH = IMAGENET_PATH
        self.imagenet_PATH = IMAGENET_PATH
        self.config = config

        if not os.path.exists(self.PATH):
            IOError('Path does not exist: ' + self.PATH)


    def get_label(self):
        dirs = os.listdir(TRAIN_PATH)
        res=[]
        for d in dirs:
            res.append(d)
        res.sort()
        return res

    def _extract_label(self):
        data = self._get_labels_from_json()
        for k, v in data.items():
            if ',' in v:
                data[k] = v.split(',')[0]
        with open(self.PATH + 'Labels.txt', 'w') as _f:
            for v in data.values():
                _f.write(v + '\n')

    def _remove_train_label(self):
        if os.path.exists(self.PATH + 'train_label/'):
            shutil.rmtree(self.PATH + 'train_label/', ignore_errors=True)
        os.makedirs(self.PATH + 'train_label/')

    def _init_train_label(self):
        data = self._get_labels_from_json()
        for k, v in data.items():
            os.makedirs(self.PATH + 'train_label/' + v + '/')
            for root, dirs, files in os.walk(self.PATH + 'train/' + k):
                for x in files:
                    shutil.copy(root + '/' + x, self.PATH + 'train_label/' + v + '/')


    def _get_labels_from_json(self):
        with open(self.config.label_json) as f:
            return json.load(f)

    def _get_image_and_label_list(self, file_pattern):
        labels = self._get_labels_from_json()
        dataset = tf.io.gfile.glob(file_pattern)

        images_list = []
        labels_list = []
        for data in dataset:
            for idx, k in enumerate(labels.keys()):
                if k in data:
                    images_list.append(data)
                    labels_list.append(idx)
                    break

        return images_list, labels_list


def get_networks(model_name,classes=1000,pretrained=True):
    if pretrained:
        return MODEL_DICT[model_name]()
    else:
        return MODEL_DICT[model_name](weights=None,classes=classes)


def create_directories():
    if os.path.exists(RANDOM_PATH):
        for root, dirs, files in os.walk(RANDOM_PATH):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
    else:
        os.makedirs(RANDOM_PATH)
    
    random_discovery_path = os.path.join(RANDOM_PATH, "random_discovery")
    if not os.path.exists(random_discovery_path):
        os.makedirs(random_discovery_path)
    for i in range(20):
        folder_name = f"random500_{i}"
        folder_path = os.path.join(RANDOM_PATH, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def random_sample_images():
   
    class_folders = [f.path for f in os.scandir(TRAIN_PATH) if f.is_dir()]


    all_images = []
    for class_folder in class_folders:
        images = [f.path for f in os.scandir(class_folder) if f.is_file()]
        all_images.extend(images)
    
    random.shuffle(all_images)

    for i in range(20):
        sample_images = all_images[i*500:(i+1)*500]
        dest_folder = os.path.join(RANDOM_PATH, f"random500_{i}")
        for img_path in sample_images:
            shutil.copy(img_path, dest_folder)

    discovery_images = random.sample(all_images, 20)
    discovery_folder = os.path.join(RANDOM_PATH, "random_discovery")
    for img_path in discovery_images:
        shutil.copy(img_path, discovery_folder)

def save_keras_model(config,model):
    tf.keras.models.save_model(
        model
        ,filepath=config.working_dir+config.teacher_model+'.h5'
        ,save_format='h5')
    
def copy_image(source_folder,destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)    

def delete_folder(destination_folder):
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)

        
if __name__=="__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    if IF_RESHUFFLE:
        create_directories()
        random_sample_images()
    print(f"Script executed successfully! A total of {20 * 500 + 20} random images were saved to {RANDOM_PATH}.")
    print("=> using pre-trained model '{}'".format(config.teacher_model))
    teacher_model_keras = get_networks(config.teacher_model)
    keras_optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    teacher_model_keras.compile(loss='categorical_crossentropy', optimizer=keras_optimizer, metrics=['accuracy'])
    save_keras_model(config, teacher_model_keras)

    dataset = imagenet(config)
    working_dir = config.working_dir

    for idx, cls in enumerate(dataset.get_label()):
        try:
            print('No.{} class name is {}'.format(idx+1, cls))
            T1 = time.time()
            config.working_dir = working_dir + cls + '/'
            model_path=working_dir+config.teacher_model+'.h5'
            config.target_class = cls
            copy_image(os.path.join(TRAIN_PATH, cls),os.path.join(RANDOM_PATH,cls))           
            ace_run.main(config,model_path)
            delete_folder(os.path.join(RANDOM_PATH,cls))
            T2 = time.time()
            print('The time is :%s s' % ((T2 - T1)))
        except Exception as e:  
            print("class {} is in error".format(cls))
            print(e)

