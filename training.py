import os 
import cv2
import joblib
import numpy as np 
import matplotlib.pyplot as plt

# Local Binary Pattern 
from skimage.feature import local_binary_pattern

# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize

# To read class from file
import csv

#util package
import cvutils

# For command line input
import argparse as ap

# warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def main(): 
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
    parser.add_argument("-l", "--imageLabels", help="Path to Image Label Files", required="True")
    args = vars(parser.parse_args())
    
    train_images = cvutils.imlist(args["trainingSet"])
    train_dic = {}
    with open(args['imageLabels'], 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            train_dic[row[0]] = int(row[1])
            
    X_test = []
    X_name = []
    y_test = []
    
    for train_image in train_images:
        # Read the image
        im = cv2.imread(train_image)
        # Convert to grayscale as LBP works on grayscale image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        radius = 3
        # Number of points to be considered as neighbourers 
        no_points = 8 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        # Calculate the histogram
        x = itemfreq(lbp.ravel())

        # Normalize the histogram
        hist = x[:, 1]/sum(x[:, 1])
        # Append image path in X_name
        X_name.append(train_image)
        # Append histogram to X_name
        X_test.append(hist)
        # Append class label in y_test
        y_test.append(train_dic[os.path.split(train_image)[1]])

    # Dump the  data
    joblib.dump((X_name, X_test, y_test), "lbp.pkl", compress=3)    
    
    
    # Display the training images (2 x 3 size)
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows,ncols)
    for row in range(nrows):
        for col in range(ncols):
            axes[row][col].imshow(cv2.imread(X_name[row*ncols+col]))
            axes[row][col].axis('off')
            axes[row][col].set_title("{}".format(os.path.split(X_name[row*ncols+col])[1]))

    # Convert to numpy and display the image
    fig.canvas.draw()
    fig.savefig('result/train_images.jpg')
    
    
if __name__ == "__main__":
    main() 
    
