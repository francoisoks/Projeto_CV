# arquivo main.py
import Load
import cv2
import numpy as np
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm

if __name__ == '__main__':
   
    DATASET='data_set'
    CLASSE_DIR=['Negative','Positive']
    STRUCT  = True
    BATCH_SIZE=4

    
    CLASSES_NUM=2

    obj_dataloader = Load.Dataloader(DATASET, batch_size=BATCH_SIZE, size=512, shuffle=True, description=True)
    
    if STRUCT==False:
        obj_dataloader.reorganize_dataset(CLASSE_DIR, CLASSE_DIR, 0.2)

    train_dataloader = obj_dataloader.get_train_dataloader()
    val_dataloader = obj_dataloader.get_val_dataloader()
    test_dataloader = obj_dataloader.get_test_dataloader()
    
    model = Load.models.MyModel(num_classes=CLASSES_NUM)
    criterio = nn.CrossEntropyLoss()
 
    otimizador = optim.Adam(model.parameters(),lr=0.00001)
    
    # EXIBE OS DADOS
    EPOCAS=30
    for epoca in range(EPOCAS):
        image_view=[]
        model.train()
        list_loss=[]
        train_sample = tqdm(train_dataloader)

        for image, target  in train_sample:
            
            otimizador.zero_grad()
            output = model(image)
            loss = criterio(output, target)
            loss.backward()
            otimizador.step()
            list_loss.append(loss.item())

            train_sample.set_description(f' Epoca {epoca+1}/{EPOCAS}, loss: {np.mean(list_loss):0.8f}')
        model.eval()

        #val_samples = tqdm(val_dataloader)
        
        #for image, mask in val_samples:
       #     pass