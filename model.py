from custom_layers import Patches, PatchEncoder, mlp
from tensorflow.keras import layers
from tensorflow import keras
from utils import config


num_patches = (config["image_size"] // config["patch_size"]) ** 2

def Vit_model(config=config):
    """
    
    retruns: Vit model 
    """
    inputs = layers.Input(shape=config["input_shape"])
    # Create patches
    patches = Patches(config["patch_size"])(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, config["projection_dim"])(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(config["transformer_layers"]):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=config["num_heads"], key_dim=config["projection_dim"], dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=config["transformer_units"], dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=config['mlp_head_units'], dropout_rate=0.3)

    bounding_box = layers.Dense(4)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)

model=Vit_model()
model.summary()