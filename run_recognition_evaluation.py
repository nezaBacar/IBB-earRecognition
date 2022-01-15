import cv2
import numpy as np
import glob
import os
import json
from pathlib import Path
from scipy.spatial.distance import cdist 
from preprocessing.preprocess import Preprocess
from metrics.evaluation_recognition import Evaluation
from torchvision import datasets, transforms, models
import torch

class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def clean_file_name(self, fname):
        return fname.split('/')[1].split(' ')[0]

    def get_annotations(self, annot_f):
        d = {}
        with open(annot_f) as f:
            lines = f.readlines()
            for line in lines:
                (key, val) = line.split(',')
                # keynum = int(self.clean_file_name(key))
                d[key] = int(val)
        return d

    def run_evaluation(self):

        eval = Evaluation()
        preprocess = Preprocess()
        
        # load and apply preprocessing to test data
        test_transforms = preprocess.preprocess_transforms()
        test_data = datasets.ImageFolder(self.images_path, transform=test_transforms)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        classes = testloader.dataset.classes
        device = torch.device("cpu")

        import nn.resnet18.classifier as resnet18_cl
        resnet18 = resnet18_cl.Classifier()

        import nn.resnet34.classifier as resnet34_cl
        resnet34 = resnet34_cl.Classifier()

        import nn.resnet50.classifier as resnet50_cl
        resnet50 = resnet50_cl.Classifier()


        y = []
        
        resnet18_arr = []
        resnet34_arr = []
        resnet50_arr = []
        
        with torch.no_grad():
          for i, (inputs, labels) in enumerate(testloader, 0):
            input, label = inputs.to(device), labels.to(device)
            actual_class = classes[label.item()-1]
            y.append(actual_class)
            
            prediction2 = resnet18.classify(input)
            resnet18_arr.append(prediction2)

            prediction3 = resnet34.classify(input)
            resnet34_arr.append(prediction3)

            prediction4 = resnet50.classify(input)
            resnet50_arr.append(prediction4)

            #image_file, _ = testloader.dataset.samples[i]
            #if (predicted_class2 == actual_class):
            #  print("{}, {}, {}, {}\n".format(image_file, predicted_class2, actual_class, i))
        
        rank1_arr = []
        rank5_arr = []

        r1 = eval.compute_rank1(y, resnet18_arr, classes)
        print('Resnet18 Rank-1[%]', r1, '%')
        rank1_arr.append(r1)

        r1_ = eval.compute_rank5(y, resnet18_arr, classes)
        print('Resnet18 Rank-5[%]', r1_, '%')
        rank5_arr.append(r1_)

        r2 = eval.compute_rank1(y, resnet34_arr, classes)
        print('Resnet34 Rank-1[%]', r2, '%')
        rank1_arr.append(r2)

        r2_ = eval.compute_rank5(y, resnet34_arr, classes)
        print('Resnet34 Rank-5[%]', r2_, '%')
        rank5_arr.append(r2_)

        r3 = eval.compute_rank1(y, resnet50_arr, classes)
        print('Resnet50 Rank-1[%]', r3, '%')
        rank1_arr.append(r3)

        r3_ = eval.compute_rank5(y, resnet50_arr, classes)
        print('Resnet50 Rank-5[%]', r3_, '%')
        rank5_arr.append(r3_)
        
        eval.plot_histogram(rank1_arr, "Rank1", "rank1 [%]", ["Resnet18", "Resnet34", "Resnet50"])
        eval.plot_histogram(rank5_arr, "Rank5", "rank5 [%]", ["Resnet18", "Resnet34", "Resnet50"])


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()