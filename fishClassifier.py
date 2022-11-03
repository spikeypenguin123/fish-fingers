import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torchvision.transforms as transforms


# convlutional block with batchnorm and max pooling
def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    
# CNN with residual connections
class FishResNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),
                                        nn.Linear(512 * 4 * 4, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out # add residual
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out # add residual
        out = self.classifier(out)
        return out

class FishClassifier:

    def __init__(self, path_to_weights, min_confidence=0.3):
        self.model = FishResNet(3, 6)
        self.model.load_state_dict(torch.load(path_to_weights))
        self.model.eval()
        self.classes = ["Carangidae", "Dinolestidae", "Enoplosidae", "Girellidae", "Microcanthidae", "Plesiopidae"]
        self.min_confidence = min_confidence

    def predict(self, image):
        resized = cv2.resize(image, (128,128))
        transform = transforms.ToTensor()
        tensor_image = transform(resized)
        with torch.no_grad():
            y_pred = self.model(tensor_image.unsqueeze(0))
            confidence_scores = nn.functional.softmax(y_pred, dim=1)
            _, y_pred_tag = torch.max(y_pred, dim = 1)
            if confidence_scores[0][y_pred_tag] < self.min_confidence:
                return None
            else:
                if y_pred_tag == 5:
                    y_pred_tag = 0
                return self.classes[y_pred_tag]