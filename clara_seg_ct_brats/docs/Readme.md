# Description
A pre-trained model for volumetric (3D) segmentation of brain tumors from multimodal MRIs based on BraTS 2018 data.

# Model Overview
## Data
The model is trained to segment 3 nested subregions of primary (gliomas) brain tumors:
the "enhancing tumor" (ET), the "tumor core" (TC), the "whole tumor" (WT) based on 4 input MRI scans (T1c, T1, T2, FLAIR).
The ET is described by areas that show hyper intensity in T1c when compared to T1,
but also when compared to "healthy" white matter in T1c.
The TC describes the bulk of the tumor, which is what is typically resected.
The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor.
The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED),
which is typically depicted by hyper-intense signal in FLAIR.

The dataset is available at "Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018."  The provided labelled data
was partitioned, based our own split, into training (243 studies) and validation (42 studies) datasets.

For more detailed description of tumor regions,
please see the Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018 data page at 
https://www.med.upenn.edu/sbia/brats2018/data.html.

## Training configuration
This model utilized a similar approach described in 3D MRI brain tumor segmentation
using autoencoder regularization, which was a winning method in
BraTS2018 [1].

The provided training configuration required 16GB GPU memory.

Model Input Shape: 224 x 224 x 128

Training Script: train.sh

## Input and output formats
Input: 4 channel 3D MRIs (T1c, T1, T2, FLAIR)

Output: 3 channels of tumor subregion 3D masks

## Scores
The model was trained with 285 cases with our own split, as shown in the datalist json file in config folder. 
The achieved Dice scores on the validation data are: 
1. Tumor core (TC): 0.851
1. Whole tumor (WT): 0.903
1. Enhancing tumor (ET): 0.773

# Availability
In order to access this model, please apply for general availability access at
https://developer.nvidia.com/clara

This model is usable only as part of Transfer Learning & Annotation Tools in Clara Train SDK container.
You can download the model from NGC registry as described in Getting Started Guide.

# Disclaimer
The content of this model is only an example.  It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. 

# Reference
[1] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." International MICCAI Brainlesion Workshop. Springer, Cham, 2018. https://arxiv.org/abs/1810.11654.
