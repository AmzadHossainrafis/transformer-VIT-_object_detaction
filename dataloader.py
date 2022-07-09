
from tensorflow.keras.utils import Sequence
import math
import numpy as np
from utils import config,x_train,y_train

class Dataloader(Sequence):
    def __init__(self,batch_size,train_data,labels):
        self.batch_size=batch_size
        self.train_data=train_data
        self.labels=labels

    def __len__(self):
        return math.ceil(len(self.train_data)/self.batch_size)

    def __getitem__(self, index):
        img_batch=self.train_data[index*self.batch_size:(index+1)*self.batch_size]
        label_batch=self.labels[index*self.batch_size:(index+1)*self.batch_size]
        #x = np.zeros((self.batch_size,config['image_size'],config['image_size'],3),dtype="float32")
        #y = np.zeros((self.batch_size), dtype="float32")
        # y=[]
        # for i in range(self.batch_size):
        #     x[i]= img_batch[i]
        #     #y[i]= label_batch[i]
        #     y.append(label_batch[i] )
        # #return img_batch ,label_batch 
        # #convert to y to tensor
        x=np.array(img_batch).astype("float32")
        y=np.array(label_batch)
        return x, y

        
if __name__ == '__main__':
    Dataloader=Dataloader(batch_size=32,train_data=x_train,labels=y_train)
    #chack the x y shape
    print(Dataloader[0][0][10].shape)
    #print(Dataloader[0][1])
    # chack the x y type 
    print(Dataloader[0][0].dtype)
    #print(Dataloader[0][1].dtype)
