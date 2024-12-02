# arquivo main.py
import Load
from metric import Metrics
from early import EarlyStop
from tensor import Log
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
    EPOCAS=200
    
    

    obj_dataloader = Load.Dataloader(DATASET, batch_size=BATCH_SIZE, size=512, shuffle=True, description=True)
    
    if STRUCT==False:
        obj_dataloader.reorganize_dataset(CLASSE_DIR, CLASSE_DIR, 0.2)

    train_dataloader = obj_dataloader.get_train_dataloader()
    val_dataloader = obj_dataloader.get_val_dataloader()
    test_dataloader = obj_dataloader.get_test_dataloader()


    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Load.models.MyModel().to(device)

    except:
        model = Load.models.MyModel().to(device)

 
    criterio = nn.BCEWithLogitsLoss()
 
    otimizador = optim.Adam(model.parameters(),lr=0.0001)
    early_stop = EarlyStop(patience=10, delta=0.001)

    scheduler = ReduceLROnPlateau(otimizador, mode='min', factor=0.1, patience=3, verbose=True)
    
    log_train = Log(BATCH_SIZE, 'train')
    log_val = Log(BATCH_SIZE, 'val')
    
    val_metrics=Metrics()
    train_metrics=Metrics()
    result_loss=[]

    for epoca in range(EPOCAS):
        
        train_loss=[]
        train_metrics.reset()
        model.train()
        train_sample = tqdm(train_dataloader)

        for image, target  in train_sample:
            
            try:
                image, target = image.to(device), target.to(device)
            except:
                pass

            otimizador.zero_grad()
            output = model(image)
            loss = criterio(output, target)
            loss.backward()
            otimizador.step()

            train_loss.append(loss.item())
            try:
                train_preds = (output > 0.5).int().cpu().numpy()
                train_targets = target.cpu().numpy()
            except:
                train_preds = (output > 0.5).int().numpy()
                train_targets = target.numpy()

            train_metrics.add_predictions(train_targets, train_preds, loss.item())

            train_sample.set_description(f' Epoca {epoca+1}/{EPOCAS}, loss: {np.mean(train_loss):0.8f}')

 
        model.eval()
        val_metrics.reset()
        val_loss = []  

        with torch.no_grad():
            for image, target in val_dataloader:
                log_val.log_image(image, epoca, path="Validation Images")

                try:
                    image, target = image.to(device), target.to(device)
                except:
                    pass
                output = model(image)
                loss = criterio(output, target)
                val_loss.append(loss.item())
                try:
                    val_preds = (output > 0.5).int().cpu().numpy()
                    val_targets = target.cpu().numpy()
                except:
                    val_preds = (output > 0.5).int().numpy()
                    val_targets = target.numpy()
                val_metrics.add_predictions(val_targets, val_preds, loss.item())


        train_results  = train_metrics.calc_metrics()
        val_results  = val_metrics.calc_metrics()

        result_loss.append(val_results['loss'])

        result_loss.append(val_results['loss'])

        scheduler.step(np.mean(val_results['loss']))
        
        log_train.log_metrics(train_results['loss'], epoca,  scalar_name='loss')
        log_val.log_metrics(val_results['loss'], epoca, scalar_name='loss')
        
        log_train.log_metrics(train_results['balanced_acc'], epoca, scalar_name='balanced_acc')
        log_val.log_metrics(val_results['balanced_acc'], epoca,  scalar_name='balanced_acc')
        
        log_train.log_metrics(train_results['f1_score'], epoca, scalar_name='f1_score')
        log_val.log_metrics(val_results['f1_score'], epoca,  scalar_name='f1_score')


        log_train.log_metrics(train_results['recall'], epoca, scalar_name='recall')
        log_val.log_metrics(val_results['recall'], epoca,  scalar_name='recall')

        log_train.log_hiper(early_stop.get_counter(),epoca,'Early_count')
        log_train.log_hiper(scheduler.get_last_lr()[-1],epoca,'LR')


        if early_stop.check(val_results['loss']):
            print(f"Parada antecipada na época {epoca}")
            break
         
    log_train.close()
    log_val.close()    
   