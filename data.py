import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras import*
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import *
#load_dataset
def data_loader(train_dir,image_size, batch_size,validation_split):
	train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, 
	                                                                    batch_size = batch_size,
	                                                                    image_size = (image_size,image_size),
	                                                                    shuffle = True, 
	                                                                    seed = 505,
	                                                                    validation_split=validation_split,
	                                                                    subset = "training")
	validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_dir, 
	                                                                    batch_size = batch_size,
	                                                                    image_size = (image_size,image_size),
	                                                                    shuffle = True, 
	                                                                    seed = 505,
	                                                                    validation_split=validation_split,
	                                                                    subset = "validation")
	AUTOTUNE = tf.data.experimental.AUTOTUNE

	train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
	validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
	return train_dataset,validation_dataset