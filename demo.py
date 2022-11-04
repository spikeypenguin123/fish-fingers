import zipfile
from PIL import Image
from fishDetector import FishDetector
from fishClassifier import FishClassifier
import cv2
import numpy as np
import argparse
import os
from get_cloud import display_pointcloud

def increase_contrast(image):
   
   # Convert to LAB colour space
   lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
   l, a, b = cv2.split(lab)

   # Apply CLAHE to L-channel only
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   adjusted_l = clahe.apply(l)

   # Merge the CLAHE enhanced L-channel with the a and b channels
   new_lab = cv2.merge((adjusted_l, a, b))
   
   # Convert back to BGR colour space
   new_image = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

   return new_image


def check_area(bbox, max_area, min_area):

    def get_area(bbox):
        return (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])

    area = get_area(bbox)
    return area < max_area and area > min_area



def getNumber(filename, prefix, suffix):
            number = int(filename[len(prefix):-len(suffix)])
            return number


if __name__ == "__main__":
        
    print("Initialising detection model...")
    fish_detector = FishDetector('fish_detector.pt')

    print("Initialising classifier model...")
    fish_classifier = FishClassifier('fish_classifier.pt')

    # Get image folder name from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="mode: 'f' for full program or 'c' for classification only")
    ap.add_argument("-f", "--dir", required=True, help="image directory name")
    args = vars(ap.parse_args())
    mode = args["mode"]

    #  Full program mode
    if mode == "f":

        markers = []

        zip_filename = args["dir"]
        zip_file = zipfile.ZipFile(zip_filename)
        file_list = zip_file.infolist()

        # Sort in frame order
        file_list = sorted(file_list, key=lambda x: getNumber(x.filename, 'vid2_filtered/final', '.jpg'))

        for f in file_list:
            ifile = zip_file.open(f)
            pil_img = Image.open(ifile)
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            image = increase_contrast(image)

            bboxes = fish_detector.predict_bboxes(image)
            n_detected = len(bboxes)
            print(f'Detected {n_detected} fish')

            # Get regions of interest using fish detector
            rois = fish_detector.get_rois(image)

            # Predict fish types
            classified_bboxes = []
            labels = []
            for i, roi in enumerate(rois):
                predicted_class = fish_classifier.predict(roi)
                if check_area(bboxes[i], 50000, 5000) and predicted_class is not None:
                    classified_bboxes.append(bboxes[i])
                    labels.append(predicted_class)
                    print(f'\t{predicted_class}')

            fish_detector.display_bboxes(image, classified_bboxes, labels=labels, timer=1)
            
            # Add camera location to pointcloud map
            filename = f.filename[len('vid2_filtered/'):]
            markers.append((filename, labels))

        display_pointcloud(markers)

    # Classification only mode
    elif mode == "c":

        folder_dir = args["dir"]
        file_list = []
        for image in os.listdir(folder_dir):
            file_list.append(image)

        # Sort in frame order
        file_list = sorted(file_list, key=lambda x: getNumber(x, 'image', '.jpg'))

        for f in file_list:
            
            image = cv2.imread(folder_dir + '/' + f)
            image = increase_contrast(image)

            bboxes = fish_detector.predict_bboxes(image)
            n_detected = len(bboxes)
            print(f'Detected {n_detected} fish')

            # Get regions of interest using fish detector
            rois = fish_detector.get_rois(image)

            # Predict fish types
            classified_bboxes = []
            labels = []
            for i, roi in enumerate(rois):
                predicted_class = fish_classifier.predict(roi)
                if predicted_class is not None:
                    classified_bboxes.append(bboxes[i])
                    labels.append(predicted_class)
                    print(f'\t{predicted_class}')

            fish_detector.display_bboxes(image, classified_bboxes, labels=labels, timer=1)
            