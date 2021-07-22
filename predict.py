import tensorflow as tf 
import cv2
from tensorflow.keras.models import load_model
#define image path and model path
im_path=""
model_path=""
# load model
model = load_model(model_path)

# predict process
img = tf.keras.preprocessing.image.load_img(file, target_size=(299, 299))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
label = np.argmax(model.predict(img_array), axis=-1)[0]

print(label)