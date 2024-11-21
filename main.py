# arquivo main.py
import Load
from metric import Metrics
from early import EarlyStop

import cv2
import numpy as np
import torch.nn as nn 
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score

if __name__ == '__main__':
   
    DATASET='data_set'
    CLASSE_DIR=['Negative','Positive']
    STRUCT  = True
    BATCH_SIZE=8
    CLASSES_NUM=2
    EPOCAS=20

    obj_dataloader = Load.Dataloader(DATASET, batch_size=BATCH_SIZE, size=512, shuffle=True, description=True)
    
    if STRUCT==False:
        obj_dataloader.reorganize_dataset(CLASSE_DIR, CLASSE_DIR, 0.2)

    train_dataloader = obj_dataloader.get_train_dataloader()
    val_dataloader = obj_dataloader.get_val_dataloader()
    test_dataloader = obj_dataloader.get_test_dataloader()
    
    model = Load.models.MyModel(num_classes=CLASSES_NUM)
    criterio = nn.CrossEntropyLoss()
 
    otimizador = optim.Adam(model.parameters(),lr=0.0001)
    early_stop = EarlyStop(patience=3, delta=0.01)

    scheduler = ReduceLROnPlateau(otimizador, mode='min', factor=0.1, patience=2, verbose=True)

    # EXIBE OS DADOS
    
    val_metrics=Metrics(num_classes=2)
    train_metrics=Metrics(num_classes=2)

    for epoca in range(EPOCAS):
        
        train_loss=[]
        train_metrics.reset()
        model.train()
        train_sample = tqdm(train_dataloader)

        for image, target  in train_sample:
            otimizador.zero_grad()
            output = model(image)
            loss = criterio(output, target)
            loss.backward()
            otimizador.step()

            train_loss.append(loss.item())

            train_preds = torch.argmax(output, dim=1)
            train_targets = target.numpy()
            train_metrics.add_predictions(train_targets, train_preds, loss.item())

            train_sample.set_description(f' Epoca {epoca+1}/{EPOCAS}, loss: {np.mean(train_loss):0.8f}')

 
        model.eval()
        val_metrics.reset()
        val_loss = []  

        with torch.no_grad():
            for image, target in val_dataloader:
                
                output = model(image)
                loss = criterio(output, target)
                val_loss.append(loss.item())
                
                val_preds = torch.argmax(output, dim=1)
                val_targets = target.numpy()
                val_metrics.add_predictions(val_targets, val_preds, loss.item())


        train_results  = train_metrics.calc_metrics()
        val_results  = val_metrics.calc_metrics()
        scheduler.step(np.mean(val_results['loss']))
                      
        print(
                f"Época {epoca + 1}/{EPOCAS}: "
                f"\nTrain Loss: {train_results['loss']:.4f}, Train Balanced Acc: {train_results['balanced_acc']:.4f}, "
                f"\nTrain F1: {train_results['f1_score']:.4f}, Train Recall: {train_results['recall']:.4f} | "
                f"\nVal Loss: {val_results['loss']:.4f}, Val Balanced Acc: {val_results['balanced_acc']:.4f}, "
                f"\nVal F1: {val_results['f1_score']:.4f}, Val Recall: {val_results['recall']:.4f}"
            )
        if early_stop.check(val_results['loss']):
            print(f"Parada antecipada na época {epoca}")
            break  

   