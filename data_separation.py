# TO APPROPRIATELY FORMAT THE DATASET FOR PYTORCH'S IMAGEFOLDER DATASET

from copy import copy
import os
import csv
import shutil

DATASET = "data/yolo4_ears/"

def copy_image(dir):
  if not (os.path.isdir(DATASET + dir + id)):
    os.mkdir(DATASET + dir + id)
  im = DATASET + name
  if os.path.exists(im):
    shutil.copy(im, DATASET + dir + id)

if __name__ == "__main__":

  with open(DATASET + "annotations/recognition/ids.csv", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
      string_list = row[0].split(",")
      name = string_list[0]
      id = string_list[1]
      if ("test" in name):
        copy_image("test_pytorch/")
      if ("train" in name):
        copy_image("train_pytorch/")