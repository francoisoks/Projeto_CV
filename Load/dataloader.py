import os
import shutil
import random
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# arquivo dataloader.py
class Data(Dataset):
    def __init__(self, image_dir: str, split: str, transform=None) -> None:
        self._image_dir = image_dir
        self._image_path = glob(f'{image_dir}/{split}/**/*.jpg', recursive=True)
        self._transform = transform
        self._split = split 
        self._targets, self._class_to_idx = self._get_targets() 
             
    def __len__(self) -> int:
        # retornar a quantidade de dados
        return len(self._image_path)

    def __getitem__(self, idx: int) -> tuple:
        # retornar image/labels/boox/mask...
        image_path = self._image_path[idx]
        image = cv2.imread(image_path)
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target=self._targets[idx]
         
        if self._transform:
            augmented_image  = self._transform(image=image)['image']
            
        augmented_image = augmented_image.float()  # Converte para float32
    
        return augmented_image,  torch.tensor(target, dtype=torch.long)
    
    def _get_targets(self) -> list:
        targets = []
        class_names = sorted({os.path.basename(os.path.dirname(img)) for img in self._image_path})
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        for image in self._image_path:
            class_name = os.path.basename(os.path.dirname(image))
            targets.append(class_to_idx[class_name])
        
        return targets, class_to_idx
  

class Dataloader:
    def __init__(self, dir:str, batch_size: int, shuffle: bool, size: int, subset: int = 0, description: bool = False) -> None:
        # construtor do dataloader
        self._dir=dir
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._size = size
        self._subset = subset
        self._description = description
        self._transform = self.compose()
     
    # criar essa estrutura data_set/treino/normal, data_set/treino/hplory e data_set/teste/normal, data_set/teste/hplory a partir de class_dir
    def reorganize_dataset(self,class_dirs: list, class_labels: list, split_train: float) -> None:
        url_train=f'{self._dir}/train'
        url_test=f'{self._dir}/test'

        if len(class_dirs) != len(class_labels):
            raise ValueError("Quantidade de Diretórios diferente da quantedade de classes")

        # Criar nova estrutura
        for label in class_labels:
            os.makedirs(f'{self._dir}/train/{label}', exist_ok=True)
            os.makedirs(f'{self._dir}/test/{label}', exist_ok=True)

        # Mover imagens 
        for i, class_dir in enumerate(class_dirs):
            # pega as imagens e embaralha 
            images = glob(f'{self._dir}/{class_dir}/*.jpg')
            random.shuffle(images)  

            # Dividir as imagens em treino e teste
            split = int(len(images) * split_train)
            train_images = images[:split]
            test_images = images[split:]
        
            for img_path in train_images:
                shutil.move(img_path, f'{url_train}/{class_labels[i]}')

            for img_path in test_images:
                shutil.move(img_path, f'{url_test}/{class_labels[i]}')
            # Remover pasta original
            if not os.listdir(f'{url_train}/{class_dir}'):
               os.rmdir(class_dir)
        print("Nova estrutura criada")

    def compose(self) -> dict: 
        process_test = A.Compose([
            A.Resize(height=self._size,width=self._size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),# artigo 1
            ToTensorV2()
        ])
        process_train = A.Compose([
            A.RandomScale(scale_limit=0.2), # Artigo 1
            A.Resize(height=self._size,width=self._size),
            A.Rotate(limit=45,p=0.2), # artigo 1
            A.VerticalFlip(p=0.2), # artigo 1
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),# artigo 1
            #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),# artigo 3
            #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1),# artigo 3
            ToTensorV2()

        ])
        return {
            'train':process_train,
            'test':process_test,
            'val':process_test
        }
    

    def get_dataloader(self, split: str) -> DataLoader:
        # retornar o dataloader baseado no split
        if split=='val':
            split='test'
        dataset = Data(self._dir, split, self._transform[split] ) # criando uma instancia de Data 
        if self._subset:
            dataset = Subset(dataset, range(self._subset))
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=self._shuffle)
        
        if self._description:
            pass
      
        
        return dataloader

    def get_train_dataloader(self) -> DataLoader: return self.get_dataloader('train')
    def get_val_dataloader(self) -> DataLoader: return self.get_dataloader('val')
    def get_test_dataloader(self) -> DataLoader: return self.get_dataloader('test')

