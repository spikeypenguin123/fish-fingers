import zipfile
from PIL import Image
from fishDetector import FishDetector
from fishClassifier import FishClassifier
import cv2
import numpy as np
import argparse
import os

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


def filter_bboxes(bboxes, min_area, max_area):

    def check_area(bbox):

        def get_area(bbox):
            return (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])

        area = get_area(bbox)
        return area < max_area and area > min_area

    return list(filter(check_area, bboxes))


def getNumber(filename, prefix, suffix):
            number = int(filename[len(prefix):-len(suffix)])
            return number


if __name__ == "__main__":
        
    print("Initialising detection model...")
    fish_detector = FishDetector('fish_detector.pt')

    print("Initialising classifier model...")
    fish_classifier = FishClassifier('fish_classifier.h5')

    # Get image folder name from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="mode: 'f' for full program or 'c' for classification only")
    ap.add_argument("-f", "--dir", required=True, help="image directory name")
    args = vars(ap.parse_args())
    mode = args["mode"]

    #  Full program mode
    if mode == "f":

        zip_filename = args["folder"]

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
            bboxes = filter_bboxes(bboxes, 50000, 5000)
            n_detected = len(bboxes)
            print(f'Detected {n_detected} fish')

            # Get regions of interest using fish detector
            rois = fish_detector.get_rois(image)

            # Predict fish types
            labels = []
            for roi in rois:
                predicted_class, confidence_score = fish_classifier.predict(roi)
                labels.append(predicted_class)
                print(predicted_class, confidence_score)

            fish_detector.display_bboxes(image, bboxes, labels=labels, timer=1)
            
            # TODO: get camera location of current image frame and add to pointcloud map

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
            # bboxes = filter_bboxes(bboxes)
            n_detected = len(bboxes)
            print(f'Detected {n_detected} fish')

            # Get regions of interest using fish detector
            rois = fish_detector.get_rois(image)

            # Predict fish types
            labels = []
            for roi in rois:
                predicted_class, confidence_score = fish_classifier.predict(roi)
                labels.append(predicted_class)
                print(predicted_class, confidence_score)

            fish_detector.display_bboxes(image, bboxes, labels=labels, timer=1)
            