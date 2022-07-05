from utils import config ,x_train,y_train,x_test,y_test
from model import Vit_model 
from dataloader import Dataloader
import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger



train_ds=Dataloader(batch_size=32,train_data=x_train,labels=y_train)
val_ds=Dataloader(batch_size=32,train_data=x_test,labels=y_test)



model=Vit_model()
optimizer = tfa.optimizers.AdamW(
        learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])

callbacks=[tf.keras.callbacks.ModelCheckpoint(
        config['check_dir']+'/checkpoint.h5',
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )]
model.compile(optimizer=optimizer, loss="mse")

model.fit(train_ds,epochs=config['epochs'],callbacks=callbacks,validation_data=val_ds)

