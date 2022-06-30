from utils import read_yaml
from model import Vit_model 
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


model=Vit_model()
model.compile(optimizer="adam", loss="mse")

model.fit()

