import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import warnings
from PIL import Image
import argparse
import json
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')
parser.add_argument('--input', default='./test_images/cautleya_spicata.jpg', action="store", type = str, help='image path')
parser.add_argument('--model', default='./model_2.h5', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='label_map.json', help='mapping the categories to real names')



arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names


def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(image, (224, 224))
    img= img/255
    img= img.numpy()
    
    return img


def predict(image_path, model_path, topk):
    
    img = Image.open(image_path)
    test_image = np.asarray(img)
    transform_img = process_image(test_image)
    redim_img = np.expand_dims(transform_img, axis=0)
    prob_pred = model_2.predict(redim_img)
    prob_pred = prob_pred.tolist()
    probs, classes = tf.math.top_k(prob_pred, k=topk)
    probs = probs.numpy().tolist()[0]
    classes = classes.numpy().tolist()[0]
    
    return probs, classes

if __name__== "__main__":

    print ("start Prediction ...")
    
    
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    
    class_names_new = dict()
    for key in class_names:
        class_names_new[str(int(key)-1)] = class_names[key]
    
    model_2 = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    probs , classes = predict(image_path, model_path, topk)
    lable_names = [class_names_new[str((key))]for key in classes]
    
    print('probs: ', probs)
    print('Classes: ',classes)
    print('Label_names: ',lable_names) 
    
    print ("end Prediction ...")
