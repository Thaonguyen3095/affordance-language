import os
import csv
import json
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Generate embeddings from images of YCB objects

#-----LOAD RESNET MODEL-----#
model = models.resnet101(pretrained=True)
# Access average pooling layer in network
model_avgpool = nn.Sequential(*list(model.children())[:-1])
model_avgpool.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#-----USE MODEL-----#
with open('../data/ycb-object-embedding.csv', 'w', newline='') as out:
    csvwriter = csv.writer(out, delimiter=',')
    #subtitute with your own path to the YCB images
    #we selected 1 front-view image to represent each object
    dir = '/home/thao/Downloads/ycb'
    for f in os.listdir(dir):
        label = f.split('_')[1:]
        l = ''
        for word in label:
            l += word + ' '
        image_label = l[:-1]
        path = os.path.join(dir, f)
        for i in os.listdir(path):
            name = i.split('.')
            if name[-1] == 'jpg':
                input_image = Image.open(os.path.join(path, i))
                input_tensor = preprocess(input_image)
                input_batch = input_tensor.unsqueeze(0)

                #move the input and model to GPU for speed if available
                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')
                    model_avgpool.to('cuda')

                with torch.no_grad():
                    try:
                        output = model_avgpool(input_batch)
                    except:
                        print(os.path.join(dir, f))
                    output = torch.flatten(output, 1)
                    csvwriter.writerow((image_label, output[0].tolist(), os.path.join(path, i)))
