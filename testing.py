import os 
import cv2
import joblib
import numpy as np 
import matplotlib.pyplot as plt

from pathlib import Path

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
    parser.add_argument("-t", "--testingSet", help="Path to Testing Set", required="True")
    parser.add_argument("-l", "--imageLabels", help="Path to Image Label Files", required="True")
    args = vars(parser.parse_args())
    
    X_name, X_test, y_test = joblib.load("lbp.pkl")
    
    # Store the path of testing images in test_images
    test_images = cvutils.imlist(args["testingSet"])
    # Dictionary containing image paths as keys and corresponding label as value
    test_dic = {}
    with open(args["imageLabels"], 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            test_dic[row[0]] = int(row[1])

    # Dict containing scores
    results_all = {}
    
    for test_image in test_images:
        path = Path(test_image)
        print("\nCalculating Normalized LBP Histogram for {}".format(path))
        # Read the image
        im = cv2.imread(str(path))
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
        # Display the query image
        results = []
        
        # For each image in the training dataset
        # Calculate the chi-squared distance and the sort the values
        for index, x in enumerate(X_test):
            score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.HISTCMP_CHISQR)
            results.append((X_name[index], round(score, 3)))
        results = sorted(results, key=lambda score: score[1])
        results_all[str(path)] = results
        print("Displaying scores for {} ** \n".format(path.stem))
        for image, score in results:
            print("{} has score {}".format(image, score))
        
    for test_image, results in results_all.items():
        path = Path(test_image)
        # Read the image
        im = cv2.imread(str(path))
        # Display the results
        nrows = 2
        ncols = 3
        fig, axes = plt.subplots(nrows,ncols)
        fig.suptitle("** Scores for -> {}**".format(path.stem))
        for row in range(nrows):
            for col in range(ncols):
                axes[row][col].imshow(cv2.imread(results[row*ncols+col][0]))
                axes[row][col].axis('off')
                axes[row][col].set_title("Score {}".format(results[row*ncols+col][1]))
        fig.canvas.draw()
        fig.savefig(f'result/{path.stem}_result.jpg')

if __name__ == "__main__":
    main() 