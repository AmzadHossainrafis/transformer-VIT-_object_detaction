from utils import  config ,data_preproses
import tensorflow as tf 
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
        return tf.convert_to_tensor(x),tf.convert_to_tensor(y) 

        
        
if __name__ == '__main__':
    x_train, y_train, x_test, y_test= data_preproses()
    dataloader=Dataloader(batch_size=256,train_data=x_train,labels=y_train)
    data=dataloader[0]
    print(data[0].shape)
    print(data[1].shape)
    #print(len(dataloader[1]))

# import matplotlib.pyplot as plt
# plt.imshow(data[0][0])