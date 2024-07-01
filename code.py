
 #importing required modules
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from patchify import patchify
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger,
ReduceLROnPlateau, EarlyStopping
""" Hyperparameters """
hp = {}
hp["image_size"] = 256
hp["num_channels"] = 3
hp["patch_size"] = 32
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"],
hp["patch_size"]*hp["patch_size"]*hp["num_channels"])
hp["batch_size"] = 8
hp["lr"] = 1e-4
hp["num_epochs"] = 10
hp["num_classes"] = 6
hp["class_names"] = ["agricultural", "airplane","baseballdiamond" ,"beach", "buildings",
"chaparral"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1
 #getting image as input and making into patches
def create_dir(path):
 if not os.path.exists(path):
 os.makedirs(path)
def load_data(path, split=0.1):
 images = shuffle(glob(os.path.join(path, "*", "*.tif")))
 split_size = int(len(images) * split)
 train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
 train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
 return train_x, valid_x, test_x
def process_image_label(path):
 """ Reading images """
 path = path.decode()
 image = cv2.imread(path, cv2.IMREAD_COLOR)
 image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
 image = image/255.0
 """ Preprocessing to patches """
 patch_shape = (hp["patch_size"], hp["patch_size"], hp["num_channels"])
 patches = patchify(image, patch_shape, hp["patch_size"])
 # patches = np.reshape(patches, (64, 25, 25, 3))
 # for i in range(64):

 # cv2.imwrite(f"files/{i}.png", patches[i])
 patches = np.reshape(patches, hp["flat_patches_shape"])
 patches = patches.astype(np.float32)
 """ Label """
 class_name = path.split("\\")[-2]
 class_idx = hp["class_names"].index(class_name)
 class_idx = np.array(class_idx, dtype=np.int32)
 return patches, class_idx
def parse(path):
 patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
 labels = tf.one_hot(labels, hp["num_classes"])
 patches.set_shape(hp["flat_patches_shape"])
 labels.set_shape(hp["num_classes"])
 return patches, labels
def tf_dataset(images, batch=32):
 ds = tf.data.Dataset.from_tensor_slices((images))
 ds = ds.map(parse).batch(batch).prefetch(8)
 return ds
 #defining vit model
class ClassToken(Layer):
 def __init__(self):
 super().__init__()
 def build(self, input_shape):
 w_init = tf.random_normal_initializer()

 self.w = tf.Variable(
 initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
 trainable = True
 )
 def call(self, inputs):
 batch_size = tf.shape(inputs)[0]
 hidden_dim = self.w.shape[-1]
 cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
 cls = tf.cast(cls, dtype=inputs.dtype)
 return cls
def mlp(x, cf):
 x = Dense(cf["mlp_dim"], activation="gelu")(x)
 x = Dropout(cf["dropout_rate"])(x)
 x = Dense(cf["hidden_dim"])(x)
 x = Dropout(cf["dropout_rate"])(x)
 return x
def transformer_encoder(x, cf):
 skip_1 = x
 x = LayerNormalization()(x)
 x = MultiHeadAttention(
 num_heads=cf["num_heads"], key_dim=cf["hidden_dim"]
 )(x, x)
 x = Add()([x, skip_1])
 skip_2 = x
 x = LayerNormalization()(x)
 x = mlp(x, cf)
 x = Add()([x, skip_2])
 return x

def ViT(cf):
 """ Inputs """
 input_shape =
(cf["num_patches"],cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
 inputs = Input(input_shape) ## (None, 256, 3072)
 """ Patch + Position Embeddings """
 patch_embed = Dense(cf["hidden_dim"])(inputs) ## (None, 256, 768)
 positions = tf.range(start=0, limit=cf["num_patches"], delta=1)
 pos_embed = Embedding(input_dim=cf["num_patches"],
output_dim=cf["hidden_dim"])(positions) ## (256, 768)
 embed = patch_embed + pos_embed ## (None, 256, 768)
 """ Adding Class Token """
 token = ClassToken()(embed)
 x = Concatenate(axis=1)([token, embed]) ## (None, 257, 768)
 for _ in range(cf["num_layers"]):
 x = transformer_encoder(x, cf)
 """ Classification Head """
 x = LayerNormalization()(x) ## (None, 257, 768)
 x = x[:, 0, :]
 x = Dense(cf["num_classes"], activation="softmax")(x)
 model = Model(inputs, x)
 return model

 config = {}
config["num_layers"] = 12
config["hidden_dim"] = 768
config["mlp_dim"] = 3072
config["num_heads"] = 12

config["dropout_rate"] = 0.1
config["num_patches"] = 256
config["patch_size"] = 32
config["num_channels"] = 3
config["num_classes"] = 5
model = ViT(config)
model.summary()
#defining Supervised contrastive loss
import torch
import torch.nn as nn
import torch.nn.functional as F
class SupervisedContrastiveLoss(nn.Module):
 def __init__(self, temperature=0.5):
 super(SupervisedContrastiveLoss, self).__init__()
 self.temperature = temperature
 def forward(self, features, labels):
 batch_size = features.shape[0]
 # Normalize features
 features = F.normalize(features, dim=1)
 # Compute cosine similarity between features
 similarity_matrix = torch.matmul(features, features.T)
 # Remove diagonal elements since we don't want them to contribute to the loss
 mask = torch.eye(batch_size, dtype=torch.bool)

 similarity_matrix = similarity_matrix.masked_fill(mask, 0)
 # Get positive and negative examples
 positive_samples_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
 negative_samples_mask = ~positive_samples_mask
 positive_samples = similarity_matrix[positive_samples_mask].view(batch_size, -1)
 negative_samples = similarity_matrix[negative_samples_mask].view(batch_size, -1)
 # Compute logits and labels for the loss
 logits = torch.cat([positive_samples, negative_samples], dim=1)
 labels = torch.zeros(batch_size, dtype=torch.long).to(labels.device)
 # Compute supervised contrastive loss
 logits /= self.temperature
 loss = F.cross_entropy(logits, labels)
 return loss
 """ Seeding """
np.random.seed(42)
tf.random.set_seed(42)
""" Directory for storing files """
dir("files")
""" Paths """
dataset_path = "/home/mp124003300/dataset3"

model_path = os.path.join("files", "model.h5")
csv_path = os.path.join("files", "log.csv")
""" Dataset """
train_x, valid_x, test_x = load_data(dataset_path)
print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")
train_ds = tf_dataset(train_x, batch=hp["batch_size"])
valid_ds = tf_dataset(valid_x, batch=hp["batch_size"])
""" Model """
model = ViT(hp)
model.compile(
 loss=SupervisedContrastiveLoss(),
 optimizer=tf.keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
 metrics=["acc"]
)
callbacks = [
 ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10,
verbose=1),
 CSVLogger(csv_path),
 EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
]
model.fit(
 train_ds,
 epochs=hp["num_epochs"],
 validation_data=valid_ds,
 callbacks=callbacks
)
#testing the test dataset and displaying acc result
test_ds = tf_dataset(test_x, batch=hp["batch_size"])
result=model.evaluate(test_ds)
print("acc:",result[0],",loss:",result[1]) 
