import os
import csv

file_name = '../data/map_clsloc.txt'
labels = '/home/thao/Downloads/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
folder = '/home/thao/Downloads/ILSVRC2012_img_val'

objects = {}
with open(file_name) as filename:
    for line in filename:
        label = str(line.split()[1])
        obj = str(line.split()[2]).split('_')
        object = ''
        for o in obj:
            object += o + ' '
        objects[label] = object[:-1]

images = os.listdir(folder)
images.sort()
image_label = {}
with open(labels) as f:
    for i, line in enumerate(f):
        image = images[i]
        label = line[:-1]
        image_label[image] = objects[label]

with open('../data/image_label.json', 'w+') as f:
    f.write(json.dumps(image_label))
