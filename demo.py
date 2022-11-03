import zipfile
from PIL import Image
from fishDetector import FishDetector
import cv2
import numpy as np
import argparse


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


if __name__ == "__main__":

    print("Initialising detection model...")
    fish_detector = FishDetector('fish_detector.pt')

    # Get image folder name from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, help="image folder containing frames")
    args = vars(ap.parse_args())
    zip_filename = args["folder"]

    zip_file = zipfile.ZipFile(zip_filename)
    file_list = zip_file.infolist()

    # Sort in frame order
    def getNumber(filename):
        prefix = 'vid2_filtered/final'
        suffix = '.jpg'
        number = int(filename[len(prefix):-len(suffix)])
        return number
    file_list = sorted(file_list, key=lambda x: getNumber(x.filename))

    for f in file_list:
        ifile = zip_file.open(f)
        pil_img = Image.open(ifile)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image = increase_contrast(image)

        bboxes = fish_detector.predict_bboxes(image)
        n_detected = len(bboxes)
        print(f'Detected {n_detected} fish')
        fish_detector.display_bboxes(image, bboxes, timer=5)

        if n_detected == 0:
            continue

        # Get regions of interest using fish detector
        rois = fish_detector.get_rois(image)

        # Predict fish types
        for roi in rois:

            # TODO: add fish classifier predictions here
            pass

        # TODO: get camera location of current image frame and add to pointcloud map