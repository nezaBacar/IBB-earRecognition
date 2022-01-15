import cv2
from torchvision import transforms

class Preprocess:

    def histogram_equlization_rgb(self, img):
        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)
        return img
    
    # resize and normalize
    def preprocess_transforms(self):
        return transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5506, 0.3965, 0.3385],
                                                           [0.2567, 0.2192, 0.2177])
                                      ])