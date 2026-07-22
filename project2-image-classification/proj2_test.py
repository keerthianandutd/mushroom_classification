import pandas as pd
import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
import json
from tensorflow.keras.utils import to_categorical
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Note that you can save models in different formats. Some format needs to save/load model and weight separately.
# Some saves the whole thing together. So, for your set up you might need to save and load differently.

def load_model_weights(model, weights = None):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model

def preprocess_image_cv2(img_path, img_height, img_width):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float32") / 255.0
    return img

def get_images_labels(df, classes, img_height=224, img_width=224, base_path = 'sample_test_data', num_classes=9):
    test_images = []
    test_labels = []
    # Load saved class index mapping
    with open('class_indices.json', 'r') as f:
        class_map = json.load(f)

    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row['image_path'])  # Point to mushroom_test folder
        class_name = row['label']

        if class_name not in class_map:
            print(f"Skipping unknown label: {class_name}")
            continue

        try:
            img = preprocess_image_cv2(img_path, img_height, img_width)
            test_images.append(img)
            test_labels.append(class_map[class_name])
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    test_images = np.array(test_images)
    test_labels = to_categorical(np.array(test_labels), num_classes=num_classes)  # One-hot encode labels

    return test_images, test_labels

def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Tranfer Learning Test")
    parser.add_argument('--model', type=str, default='mobilenet_v2_final.keras', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='weight file if needed')
    parser.add_argument('--test_csv', type=str, default='mushrooms_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model = args.model
    weights = args.weights
    test_csv = args.test_csv

    test_df = pd.read_csv(test_csv)
    classes = {'Agaricus', 'Amanita', 'Boletus', 'Cortinarius','Entoloma','Hygrocybe',
				'Lactarius', 'Russula', 'Suillus'}
    
    # Rewrite the code to match with your setup
    test_images, test_labels = get_images_labels(test_df, classes)
    
    my_model = load_model_weights(model)
    loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print('Test model, accuracy: {:5.5f}%'.format(100 * acc))

    