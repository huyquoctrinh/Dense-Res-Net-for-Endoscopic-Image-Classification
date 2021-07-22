#import neccessary packages
import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras import*
from tensorflow.keras.models import *
# IMG_SHAPE =(256,256,3)
#define ResDense Net
def build_model(img_size,num_classes,trainable):
	IMG_SHAPE = (img_size,img_size,3)
	model1 = tf.keras.applications.ResNet152(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
	model2 = tf.keras.applications.DenseNet201(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
	data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
	preprocess_input = tf.keras.applications.resnet.preprocess_input
	rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
	global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
	# prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')
	if num_classes <=2 :
		prediction_layer= tf.keras.layers.Dense(1,activation='sigmoid')
	else:
		prediction_layer= tf.keras.layers.Dense(num_classes,activation='softmax')
	# prediction_layer = tf.keras.layers.Dense(num_classes)
	model1.trainable= trainable
	model2.trainable= trainable
	inputs = tf.keras.Input(shape=(256, 256, 3))
	data = data_augmentation(inputs)
	data= preprocess_input(data)
	data = rescale(data)
	x = model1(data,training = False)
	y = model2(data,training = False)
	x = Conv2D(1024,kernel_size=(3,3),strides = (1,1),padding = 'same')(x)
	y = Conv2D(1024,kernel_size=(3,3),strides = (1,1),padding = 'same')(y)
	merge = Add()([x,y])
	merge = global_average_layer(merge)
	merge = tf.keras.layers.Dropout(0.2)(merge)
	outputs = prediction_layer(merge)
	model = tf.keras.Model(inputs, outputs)
	return model,model1,model2
# model,model1,model2=build_model(256,8,False)








