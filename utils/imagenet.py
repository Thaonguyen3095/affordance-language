import os
import csv

# Generates a dictionary mapping ImageNet images to their labels

file_name = '../data/map_clsloc.txt' #contains mapping between ImageNet classes and labels
#subtitute with your own path to the ILSVRC2012 validation images and image labels
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
