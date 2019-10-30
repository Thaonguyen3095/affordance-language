import os
import csv
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


#-----TEST MODEL-----#
with open('data/object-embedding.csv', 'w', newline='', encoding='latin-1') as out:
    csvwriter = csv.writer(out, delimiter=',')
    dir = '/home/thao/affordance-language/test-images'
    for f in os.listdir(dir):
        name = f.split('.')[0]
        obj, num = name.split('_')
        input_image = Image.open(dir+'/'+f)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model_avgpool.to('cuda')

        with torch.no_grad():
            output = model_avgpool(input_batch)
            output = torch.flatten(output, 1)
            csvwriter.writerow((obj, output[0].tolist(), num))
