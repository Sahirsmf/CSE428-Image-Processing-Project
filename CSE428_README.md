# CSE428 Project — Joint Pet Segmentation & Breed Classification

A computer vision project for CSE428 (Summer 26) that jointly performs pixel-level image segmentation and 37-breed classification on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), comparing a Base U-Net against an Attention U-Net, with two extended bonus experiments.

**[Open in Colab](https://colab.research.google.com/github/[github-username]/[repo-name]/blob/main/CSE428_Project.ipynb)**

## Overview

The dataset (7,349 images across 37 cat/dog breeds) is used to train encoder-decoder models that predict a foreground/background segmentation mask and a breed label from the same input image, using a shared encoder with two task-specific heads.

## What's implemented

- **Base U-Net** — joint segmentation + classification, trained end-to-end
- **Attention U-Net** — U-Net with attention gates on the skip connections, same joint setup
- **Bonus: classifier backbone comparison** — MobileNetV2, EfficientNet-B0, and DenseNet121 benchmarked as alternative classifier backbones
- **Bonus: data augmentation** — Attention U-Net retrained with augmentation to evaluate its effect on performance
- **Demo pipeline** — upload a new image and visualize the predicted segmentation mask and breed label from every trained model side-by-side against ground truth

## Dataset

- Oxford-IIIT Pet Dataset — [download link](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- 7,349 images, 37 breeds (dogs and cats)
- Split: 3,128 train / 552 validation / 3,669 test
- Pixel-level trimaps (foreground / background / boundary) converted to binary foreground/background masks for segmentation

## Model Architecture

- Shared convolutional encoder (4 downsampling stages + bottleneck)
- Segmentation decoder with skip connections (Attention U-Net adds attention gates here)
- Classifier head on the bottleneck features (global average pool → FC → dropout → FC)
- Both tasks trained jointly with a combined loss

## Evaluation Metrics

Reported across train, validation, and test splits:
- **Segmentation:** mean IoU (mIoU), Dice coefficient, pixel accuracy
- **Classification:** accuracy, macro precision, macro recall, macro F1

## Tech Stack

PyTorch · torchvision · scikit-learn · pandas · NumPy · Matplotlib · PIL · Google Colab (GPU runtime + Drive for checkpoint storage)

## Running the Notebook

This project was built and run entirely in Google Colab, so the notebook is the only file needed:

1. Open `CSE428_Project.ipynb` in Colab (badge link above, or upload manually).
2. Set **Runtime → Change runtime type → GPU**.
3. Run the setup/config cell — it will mount your Google Drive to store model checkpoints and metric reports.
4. Set `ACTIVE_STAGE` in the config cell to the stage you want to run, then run all cells:
   - `"base"` — train the Base U-Net
   - `"attention"` — train the Attention U-Net
   - `"mobilenet"` / `"efficientnet"` / `"densenet"` — train each bonus classifier backbone
   - `"augmentation"` — train the augmented Attention U-Net
   - `"final"` — generate the comparison tables across all trained models (requires the checkpoints above to already exist)
   - `"demo"` — load all saved checkpoints and run predictions on an image you upload

## Results

`[Add your final mIoU / Dice / pixel accuracy and classification accuracy / precision / recall / F1 numbers here once you've run the "final" stage — pull them straight from the report CSVs saved to your Drive artifact folder.]`

## Acknowledgments

- Dataset: O. M. Parkhi et al., *Cats and Dogs*, Oxford-IIIT Pet Dataset
- U-Net: Ronneberger et al., [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Attention U-Net: Oktay et al., [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)
- Project completed for CSE428, Summer 2026
