# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# # # Vibhor
# # %pip install pynvml
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Projects/EECS571/REDD/projectRedd
# %cd /content/drive/MyDrive/REDD/projectRedd

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir tf_logs_QUANT/

# Commented out IPython magic to ensure Python compatibility.
import json
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
# from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import os 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.transforms.functional import crop
import torchvision.transforms.functional as F
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision import tv_tensors
from torchvision.io import read_image

from PIL import Image,ImageDraw
from torch.utils.data import Dataset
from pathlib import Path

from itertools import chain
import cv2
from skimage import io, transform
from tqdm import tqdm

from pre_trained_PSPNet.ptsemseg.pspnet import pspnet

from torch.utils.data.sampler import SubsetRandomSampler
import segmentation_models_pytorch as smp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

from collections import OrderedDict
def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state

    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


# PSPNet to Quantisized model
class QuantizablePSPNet(nn.Module):
    def __init__(self,model, *args, **kwargs):
        super(QuantizablePSPNet, self).__init__(*args, **kwargs)
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

from pre_trained_PSPNet.ptsemseg.utils import conv2DBatchNormRelu

def fuse_model(model):
    for m in model.modules():
        if type(m) == conv2DBatchNormRelu:
          torch.ao.quantization.fuse_modules(m.cbr_unit, ['0', '1', '2'], inplace=True)

# model = QuantizablePSPNet(psp_model)
# model.eval()
# model.to("cpu")

# fuse_model(model)

# qconfig = torch.ao.quantization.get_default_qconfig('qnnpack') # or 'qnnpack' for ARM CPUs
# model.qconfig = qconfig
# torch.ao.quantization.backend = "qnnpack"
# torch.ao.quantization.prepare(model, inplace=True)

# torch.ao.quantization.convert(model, inplace=True)

# torch.save(model.state_dict(),os.path.join("models_weights", '_'.join(["PSPNet_quantized_new","new_date.pth"])))

def get_disease_color_map():
    disease_color_map = {
    'Atelectasis': (1,(255, 0, 0)),         # Red
    'Calcification': (2,(0, 255, 0)),       # Green
    'Cardiomegaly': (3,(0, 0, 255)),        # Blue
    'Consolidation': (4,(255, 255, 0)),     # Yellow
    'Diffuse Nodule': (5,(255, 0, 255)),    # Magenta
    'Effusion': (6,(0, 255, 255)),          # Cyan
    'Emphysema': (7,(128, 0, 0)),           # Dark Red
    'Fibrosis': (8,(0, 128, 0)),            # Dark Green
    'Fracture': (9,(0, 0, 128)),            # Dark Blue
    'Mass': (10,(128, 128, 0)),              # Olive
    'Nodule': (11,(128, 0, 128)),            # Purple
    'Pleural Thickening': (12,(0, 128, 128)), # Teal
    'Pneumothorax': (13,(128, 128, 128))     # Gray
    }
    return disease_color_map

# Function to create masks of different colors for each disease category
def create_channel_masks(image_shape, polygons, syms):
    #colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              #(128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
              #(255, 128, 0)]  # 13 distinct colors for 13 disease categories
    disease_color_map = get_disease_color_map()
    mask = np.full(image_shape,0)

    for idx, (polygon, sym) in enumerate(zip(polygons, syms)):
      ch = disease_color_map[sym][0] - 1
      mask[:,:,ch] = mask[:,:,ch] | get_mask(polygon)

    return mask

def get_mask(points):
    mask = np.full((1024,1024),False)
    if len(points) ==0:
        return mask
    img = Image.new('L', (1024,1024), 0)
    poly = [(x,y) for x,y in points]
    ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
    curr_mask = np.array(img)

    return curr_mask

from torch.utils.data import Dataset

def load_image(path:str):
    image = Image.open(path)
    if image.mode !='RGB' or image.mode != 'RGBA': # converts grayscale image to RGB
        image = image.convert("RGB")
    return image

class ImageDataset(Dataset):
    def __init__(self,img_folder:str,transforms_ ,transforms_n,image_size:int,annotation_json,num_classes=13):

        self.folder = img_folder
        self.files = list(annotation_json.keys())
        self.transform = transforms_
        self.transform_n = transform_n
        self.image_size = image_size
        self.annotation_json = annotation_json
        self.nclasses = num_classes

    def __getitem__(self,index):

        img = load_image(os.path.join(self.folder,self.files[index]))
        mask = create_channel_masks((1024,1024,13),self.annotation_json[self.files[index]]["polygons"],self.annotation_json[self.files[index]]["syms"])
        # img = tv_tensors.Image(img,dtype=torch.float32)
        masks = [tv_tensors.Mask(mask[:,:,ch],dtype = torch.uint8) for ch in range(self.nclasses)]

        img, masks = self.transform(img,masks)

        return self.transform_n(img),torch.stack(masks,0)
        return img,torch.stack(masks,0)

    def __len__(self):
        return len(self.files)

from torchvision.transforms.autoaugment import InterpolationMode


train_transforms = v2.Compose([
        v2.Resize(256),
        v2.RandomRotation(5),
        # v2.RandomResizedCrop(256),
        v2.ToTensor()
        ])
transform_n = v2.Compose([v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# train_json_path = 'reformatted_ChestX_Det_train.json'
# with open(train_json_path,'r')as f:
#     train_json =  json.load(f)

# train_path = 'train_data'

# dataset = ImageDataset(img_folder=train_path,transforms_=train_transforms,transforms_n = transform_n,image_size = 512,annotation_json = train_json,num_classes =13)
# batch_size = 16
# validation_split = .1
# shuffle_dataset = True
# random_seed= 42

# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# # Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True,sampler = train_sampler)
# val_loder = torch.utils.data.DataLoader(dataset, batch_size=1,drop_last=True,sampler = valid_sampler)

def load_image(path:str):
    image = Image.open(path)
    if image.mode !='RGB' or image.mode != 'RGBA': # converts grayscale image to RGB
        image = image.convert("RGB")
    return image

"""

```
# This is formatted as code
```

# Accuracy on *Test*"""

#Load QModel and Fuse
n_classes = len(get_disease_color_map())
psp_model = pspnet(n_classes)

psp_model = QuantizablePSPNet(psp_model)
psp_model.eval()
psp_model.to("cpu")



backend = "qnnpack"
psp_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend

fuse_model(psp_model)


torch.ao.quantization.prepare(psp_model, inplace=True)
torch.ao.quantization.convert(psp_model, inplace=True)

state = torch.load(os.path.join("model_weights", '_'.join(["psp_net","quant_cal_11_26.pth"])),map_location=device)
psp_model.load_state_dict(state)
psp_model.eval()
psp_model.to(device)


"""Plotting Image"""

def tensor_2_image(image):
    return image.numpy().transpose((1, 2, 0))

def imshow(image, ax=None, title=None, normalize=False):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
def draw_segment(tensor,poly,color= (255,0,0)):
    image = tensor_2_image(tensor)
    points = np.array([poly],dtype='int32')
    cv2.fillPoly(image,pts = points,color= color)
    return image

def get_mask(points):
    mask = np.full((1024,1024),False)
    if len(points) ==0:

        return mask
    for i, poly in enumerate(points):
        img = Image.new('L', (1024,1024), 0)
        poly = [(x,y) for x,y in poly]
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        curr_mask = np.array(img)
        mask = np.dstack((mask,curr_mask))
    return mask[:,:,1:].transpose((2,0,1))
def generate_mask(data,jsonF):
    mask = []
    for d in data:
        polygons =  jsonF[d]["polygons"]
        mask.append(get_mask(polygons))
    return mask

# img_org = load_image(f'test_data/40797.png')
transform_rez = v2.Resize(256)

transform_img = v2.Compose([v2.Resize(256),v2.ToTensor(),v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# img = transform_img(img_org).unsqueeze(0)
# img_org = transform_rez(img_org)
# # img = torch.ones((1,3,256,256))
# print(f' lol {type(img)}')

# psp_model.eval()

# vout = psp_model.forward(img.to(device,dtype=torch.float))
# print("Finished")

# vout.size()

# img_org = img_org.permute(0,2,3,1)
# vout = torch.sigmoid(vout).detach().numpy()
# vout[vout>=0.5]=1
# vout[vout<0.5] =0

# plt.imshow(np.array(img_org))
# plt.show()

# # Plotting a specific channel of the output
# plt.imshow(vout[0, 6, :, :], cmap='gray')
# plt.show()

def process_image(image_path):
    # Load the image
    img_org = load_image(image_path)

    # Preprocess the image (resize, normalize, etc.)
    transform_img = v2.Compose([
        v2.Resize(256),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_img(img_org).unsqueeze(0)

    # Ensure the model is in evaluation mode
    psp_model.eval()

    # Process the image with the model
    with torch.no_grad():  # Disable gradient calculations for inference
        output = psp_model(img.to(device, dtype=torch.float))

    # Post-process the output (e.g., applying a threshold, resizing)

    output_processed = post_process_output(output)
    print("Shape of processed output:", output_processed.shape)
    save_or_display_result(output_processed, image_path)

    # Save or display the result
    save_or_display_result(output_processed, image_path)

def post_process_output(output):
    # Example post-processing
    output = torch.sigmoid(output).detach().cpu().numpy()
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    return output

# Define the directory for saving processed images
PROCESSED_IMAGE_DIR = "processed_images"

def save_or_display_result(output, original_image_path):
    try:
        # Check if the output is valid
        if output is None or len(output) == 0:
            print("Output is empty or None")
            return

        # Create the directory if it doesn't exist
        if not os.path.exists(PROCESSED_IMAGE_DIR):
            print(f"Creating directory: {PROCESSED_IMAGE_DIR}")
            os.makedirs(PROCESSED_IMAGE_DIR)

        # Extract the filename from the original image path
        filename = os.path.basename(original_image_path)

        # Create a new filename for the processed image
        processed_filename = filename.replace('.png', '_processed.png').replace('.jpg', '_processed.jpg').replace('.jpeg', '_processed.jpeg')

        # Full path for the processed image
        processed_image_path = os.path.join(PROCESSED_IMAGE_DIR, processed_filename)

        # Process the output for saving
        processed_image = output[0]
        if len(processed_image.shape) == 4 and processed_image.shape[1] == 13:
            # # Sum up all channels and squeeze out batch and channel dimensions
            # processed_image = processed_image.sum(axis=1).squeeze(0)

            # # Normalize the image for display
            # processed_image = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min())

            # # Save the processed image
            # plt.imsave(processed_image_path, processed_image, cmap='gray')
            fig= plt.figure(figsize=(10, 10))
            for i in range(13):
                ax = fig.add_subplot(4,4,i+1)
                ax.imshow(output[0,i,:,:],cmap = 'gray')
            fig.savefig(processed_image_path)
            print(f"Processed image saved to: {processed_image_path}")
        else:
            raise ValueError(f"Unexpected shape for image array: {processed_image.shape}")

    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an input scan image.")
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    if args.image_path:
        print(args.image_path)
        process_image(args.image_path)
    else:
        print("No image path provided.")

