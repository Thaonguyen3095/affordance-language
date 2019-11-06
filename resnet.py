import os
import csv
import json
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

#-----LOAD MODEL-----#

# VGG
#model = models.vgg11(pretrained=True)
#model = models.vgg11_bn(pretrained=True)
#model = models.vgg13(pretrained=True)
#model = models.vgg13_bn(pretrained=True)
#model = models.vgg16(pretrained=True)
#model = models.vgg16_bn(pretrained=True)
#model = models.vgg19(pretrained=True)
#model = models.vgg19_bn(pretrained=True)

# RESNET
#model = models.resnet18(pretrained=True)
#model = models.resnet34(pretrained=True)
model = models.resnet50(pretrained=True)
#model = models.resnet101(pretrained=True)
#model = models.resnet152(pretrained=True)

# Access average pooling layer in network
model_avgpool = nn.Sequential(*list(model.children())[:-1])
model_avgpool.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#-----TEST MODEL-----#
labels = {}
with open('data/object-embedding.csv', 'w', newline='') as out:
    with open('data/image_label.json') as data:
        csvwriter = csv.writer(out, delimiter=',')
        for line in data:
            image_label = json.loads(line)
        dir = '/home/thao/Downloads/ILSVRC2012_img_val'
        for f in os.listdir(dir):
            input_image = Image.open(os.path.join(dir, f))
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model_avgpool.to('cuda')

            with torch.no_grad():
                try:
                    output = model_avgpool(input_batch)
                except:
                    print(os.path.join(dir, f))
                output = torch.flatten(output, 1)
                csvwriter.writerow((image_label[f], output[0].tolist(), f))
