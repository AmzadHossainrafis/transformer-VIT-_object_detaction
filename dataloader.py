
from tensorflow.keras.utils import Sequence
import math
import numpy as np


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
        # x = np.zeros((self.batch_size,config['image_size'],config['image_size'],3))
        # y = np.zeros((self.batch_size) + 4, dtype="float32")
        x,y=[],[]
        for i in range(self.batch_size):
            x.append(img_batch[i])
            y.append(label_batch[i] )
        #return img_batch ,label_batch 
        return np.asarray(x,dtype=float), np.array(y)

        
