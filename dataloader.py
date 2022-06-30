from utils import  config 
import tensorflow as tf 
from tensorflow.keras.utils import Sequence
import math

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

        return img_batch,label_batch
        

        
        
if __name__ == '__main__':
    dataloader=Dataloader(batch_size=2,train_data,labels)
    print(dataloader[0].shape())
    print(dataloader[1].shape())