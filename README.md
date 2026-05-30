# Blurred Person Detection Using YOLOv8

## Overview

This Computer Vision project focuses on improving person detection performance in blurred environments using a fine-tuned YOLOv8 model.

A custom blurred-person dataset was used to train and evaluate the model. The performance of the fine-tuned model was compared with the baseline model to demonstrate the effectiveness of domain-specific training.

## Dataset

The project uses a custom blurred-person detection dataset prepared in YOLO format.

Dataset Link:

https://www.kaggle.com/datasets/ahmedrdata/blurred-person-detection-dataset-yolov8

Dataset Details:

* Training Images: 364
* Testing Images: 91
* Class: Person

The dataset was specifically designed to evaluate object detection performance in blurred visual environments.


## Model

* YOLOv8n (Pretrained)
* Fine-Tuned on a Custom Blurred-Person Dataset

## Evaluation Results

| Metric    | Value |
| --------- | ----- |
| Precision | 0.911 |
| Recall    | 0.898 |
| mAP@50    | 0.928 |
| mAP@50-95 | 0.671 |

## Files

* train.py
* validate.py
* inference.py
* gui.py

 ## Model Weights

Due to GitHub file size limitations, trained model weights may not be included in the repository.

Main Model:

* blur_finetuned.pt

 

## Results

Training curves, confusion matrix, and precision-recall curves are available in the results folder.

## Course

Computer Vision Final Project
