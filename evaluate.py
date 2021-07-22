from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
#define test_dataset directory
test_dir = " "
#load test dataset
test_datagen = ImageDataGenerator(rescale=1. / 127.5)
test_data = test_datagen.flow_from_directory(test_dir,
                                            target_size=(256, 256),
                                            batch_size=128,
                                            class_mode='categorical')
y_true = test_data.classes
#predict test_dataset
Y_pred = model.predict(test_data)


#plot confusion matrix and classfication report
cm = metrics.multilabel_confusion_matrix(test_data.classes, y_pred)
print(metrics.classification_report(test_data.classes,y_pred))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()
