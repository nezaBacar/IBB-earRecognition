import cv2
import numpy as np
from podm.podm import get_pascal_voc_metrics, BoundingBox, MetricPerClass
import os 
class Detector:
  # Load Yolo

  # Insert here the path of your images
  def detect(self, img, im_name):
    net = cv2.dnn.readNet("detectors/yolo4tiny_ear_detector/yolov4_training_last3700.weights", "detectors/yolo4tiny_ear_detector/yolov4-testing-tiny.cfg")
    classes = ["Ear"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    #output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    p = []

    width = 480
    height = 360
    for out in outs:
      for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Object detected
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        if confidence > 0.3:
          boxes.append([x, y, w, h])
          confidences.append(float(confidence))
          class_ids.append(class_id)
    return [boxes, confidences]