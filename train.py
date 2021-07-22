import tensorflow as tf
from data import *
from model import *

#define training parameter
img_size = 256
num_classes = 8
train_dir = "D:/kvasir-dataset-v2"
model_first_name="first.h5"
model_after_finetuning="after.h5"
batch_size = 16
validation_split =0.25
base_learning_rate= 0.001
epochs = 20

#define model and load_data
model,model1,model2 = build_model(img_size,num_classes,False)
train_dataset,validation_dataset= data_loader(train_dir,img_size,batch_size,validation_split)

#define loss function
if num_classes>=2:
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
else:
	loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False)

#compile model and define ckpt
model.compile(loss=loss,
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath = '/content/checkpoint', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')  


#training
history_fine = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=validation_dataset, callbacks= [checkpoint])

#save model
model.save(model_first_name)

#start finetuning
model1.trainable=True
model2.trainable= True
model.trainable= True

fine_tune_at = 100
for layer in model1.layers[:fine_tune_at]:
  layer.trainable =  False
for layer in model2.layers[:50]:
  layer.trainable =  False

model.compile(loss=loss,
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

history_fine = model.fit(train_dataset,
                         initial_epoch=history.epoch[-1],
                    epochs=14,
                    validation_data=validation_dataset, callbacks= [checkpoint])

model.save(model_after_finetuning)