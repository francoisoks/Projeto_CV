# arquivo main.py
import Load
import cv2
import numpy as np
if __name__ == '__main__':
   
    DATASET='data_set'
    CLASSE_DIR=['Negative','Positive']
    STRUCT  = True
    BATCH_SIZE=2

    obj_dataloader = Load.Dataloader(DATASET, batch_size=BATCH_SIZE, size=512, shuffle=True, description=True)
    
    if STRUCT==False:
        obj_dataloader.reorganize_dataset(CLASSE_DIR, CLASSE_DIR, 0.2)

    train_dataloader = obj_dataloader.get_train_dataloader()
    # val_dataloader = obj_dataloader.get_val_dataloader()
    # test_dataloader = obj_dataloader.get_test_dataloader()

    # EXIBE OS DADOS
    image_view=[]
    for image, target  in train_dataloader:
        for i in range(BATCH_SIZE):
            try:
                image_np = image.detach().cpu().numpy()[i].transpose(1, 2, 0)
                image_view.append([image_np,target[i]])
            except:
                print('Sua base encontra-se com alguma classe desbalanceada!')
    
    for image in image_view:
        cv2.imshow(image[1], image[0])
        if cv2.waitKey(0) == ord('q'):
            break 
        