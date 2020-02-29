import cv2
import collections
from tensorflow.keras.models import load_model
from helper.FilePickling import pkl_load
from helper.ImageTransform import load_preprocess,img2batch,extract_features
import matplotlib.pyplot as plt
import argparse
from sklearn.externals import joblib
import numpy as np

WIDTH = 384
HEIGHT = 256
BBOX = 32
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",help='path to input file')
    args = parser.parse_args()

    img_path = args.input
    
    if img_path == None:
        print('You should insert your input file. Try again..')
        input()
        exit()

    # Load pretrained model
    model = load_model("assets/models/model_val_0.8391.h5",compile=True)

    # Load scaler object
    min_ = np.load("assets/objects/min_data.npy")
    max_ = np.load("assets/objects/max_data.npy")


    # Load and preprocess data
    img = load_preprocess(img_path)

    # Split data into small batches
    batches = img2batch(img)

    batches_labels = []

    for batch in batches:
        #features.append(extract_features(batch))
        feature = np.array(extract_features(batch))
        feature = np.expand_dims(feature,0)
        feature = (feature - min_)/(max_-min_)
        y_hat = (model.predict(feature)[0][0]>0.5)*1
        batches_labels.append(y_hat)
    
    # Count number of positive class
    ones = collections.Counter(batches_labels)[1]

    if ones >= 9:
        label = "tampered"
    else:
        label = "natural"

    # Get positive map to plot
    positive_map = np.array(batches_labels).reshape(int(HEIGHT/32),int(WIDTH/32))

    img = cv2.cvtColor(img,cv2.COLOR_YCrCb2RGB)
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Original image")
    # Remove x,y axis
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,2,2)
    plt.imshow(positive_map)
    plt.title("Tampered map. Predict: {}".format(label))
    
    # Remove x,y axis
    plt.xticks([])
    plt.yticks([])

    plt.show()