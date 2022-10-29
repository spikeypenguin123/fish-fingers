# fish-fingers

## Fish Detection

Fish are detected in a given image frame using a YOLO-v5 model. To preview the detected bounding boxes for a given image frame, run the following:
``` 
python3 fishDetector.py --image <IMAGE_FILENAME>
```

To instantiate the model, import the class and use:
```
fish_detector = FishDetector('fish_detector.pt')
```

The ROI cropping functionality is provided by the ```get_rois()``` method, which returns a list of images (each representing an ROI). For further fish classification, these images should be resized to the required image size.
```
rois = fish_detector.get_rois(image)
```
