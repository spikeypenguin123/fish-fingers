import argparse
import torch
import math
import cv2
import time

# Uncomment this if ssl error occurs
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class FishDetector:

    def __init__(self, path_to_weights, confidence_threshold=0.1, bbox_colour=(0,0,255), bbox_thickness=2):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights, force_reload=True)
        self.confidence_threshold = confidence_threshold
        self.bbox_colour = bbox_colour  # in BGR
        self.bbox_thickness = bbox_thickness

    def predict_bboxes(self, image):
        image = image[..., ::-1] # convert opencv BGR format to RGB
        predictions = self.model(image)
        predictions_as_df = predictions.pandas().xyxy[0]
        thresholded_predictions = self.threshold_predictions(predictions_as_df)
        bboxes = self.get_bboxes(thresholded_predictions)
        return bboxes

    def threshold_predictions(self, predictions):

        # Only keep the predictions with a confidence score greater than the threshold
        is_above_threshold = predictions['confidence'] > self.confidence_threshold
        thresholded_predictions = predictions[is_above_threshold]
        return thresholded_predictions

    def get_bboxes(self, predictions):

        # Get bottom-left and top-right integer coords of bounding boxes
        x_min = [math.floor(x) for x in list(predictions['xmin'])]
        y_min = [math.floor(y) for y in list(predictions['ymin'])]
        x_max = [math.ceil(x) for x in list(predictions['xmax'])]
        y_max = [math.ceil(y) for y in list(predictions['ymax'])]

        # Zip into tuples
        bottom_left_coords = list(zip(x_min, y_min))
        top_right_coords = list(zip(x_max, y_max))
        bboxes = list(zip(bottom_left_coords, top_right_coords))

        return bboxes

    def display_bboxes(self, image, bboxes, labels=None, timer=None):

        count = 0
        for bbox in bboxes:
            (bottom_left_coords, top_right_coords) = bbox
            image = cv2.rectangle(image, bottom_left_coords, top_right_coords, self.bbox_colour, 2)
            if labels is not None:
                image = cv2.putText(image, labels[count], bottom_left_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, self.bbox_colour, 2, cv2.LINE_AA)
                count += 1

        cv2.imshow("Bounding Boxes", image)
        if timer is not None:
            cv2.waitKey(timer)
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_rois(self, image):

        rois = []

        # Crop each detected bounding box in the image
        bboxes = self.predict_bboxes(image)
        for bbox in bboxes:
            (bottom_left_coords, top_right_coords) = bbox
            cropped = image[bottom_left_coords[1]:top_right_coords[1], bottom_left_coords[0]:top_right_coords[0]]
            rois.append(cropped)

        return rois


if __name__ == "__main__":

    # Get image filename from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--image", required=True, help="image for bounding box prediction")
    args = vars(ap.parse_args())
    image_fname = args["image"]

    print("Initialising model...")
    fish_detector = FishDetector('fish_detector.pt')

    print("Predicting bounding boxes...")
    image = cv2.imread(image_fname)
    bboxes = fish_detector.predict_bboxes(image)
    n_detected = len(bboxes)
    print(f'Detected {n_detected} fish')
    fish_detector.display_bboxes(image, bboxes)
