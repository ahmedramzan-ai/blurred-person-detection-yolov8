from ultralytics import YOLO
import os

def train_model():
    model = YOLO("yolov8n.pt")


    
    model.train(
        data="dataset/data.yaml",

        
        epochs=150,
        imgsz=640,
        batch=16,
        lr0=0.001,

         
        weight_decay=0.0005,   
        patience=15,          
        multi_scale=True,      

       
        project="results",
        name="blur_finetune",
        save=True,
        verbose=True
    )
