# TO EXTRACT BOUNDING BOXES FROM DETECTION DATASET

import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import matplotlib.pyplot as plt


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path_detector = config['images_path']
    
    def run_evaluation(self):
        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))

        import detectors.yolo4tiny_ear_detector.ear_detector as yolo4tiny_ear_detector
        yolo4tiny_ear_detector = yolo4tiny_ear_detector.Detector()
        
        for im_name in im_list:
            # Read an image
            img = cv2.imread(im_name)
          
            pr = []
          
            prediction_list_cascade3, all_pred_3 = yolo4tiny_ear_detector.detect(img, im_name)
            pr.append(prediction_list_cascade3)
            print(prediction_list_cascade3, all_pred_3)

            if (all_pred_3):
                max_id = all_pred_3.index(max(all_pred_3)) 
                max_box = prediction_list_cascade3[max_id]
                x = max_box[0]
                y = max_box[1]
                w = max_box[2]
                h = max_box[3]
                print(x, y, w, h)
                cropped_image = img[y:y+h, x:x+w]
                plt.imshow(cropped_image)
                print("Before saving image:")  
                full_name = im_name.replace("data/ears/train/", "")
                cv2.imwrite(full_name, cropped_image)
                
if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()