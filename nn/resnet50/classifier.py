import torch 
from torchvision.models import resnet50
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os

class Classifier:
	# INITIATE THE MODEL AND LOAD THE CHECKPOINT
	def __init__(self, current_folder = os.path.dirname(os.path.abspath(__file__))):
		self.device = torch.device("cpu")
		#self.annot_folder = os.path.join(current_folder, 'classifier_resnet50_50_jitter_crop.pt') #28%
		self.annot_folder = os.path.join(current_folder, 'classifier_resnet50_30_custom.pt') #22.8%
		self.checkpoint = torch.load(self.annot_folder, map_location='cpu')
    
		self.model = resnet50(pretrained=True)
		self.model.fc = torch.nn.Linear(self.model.fc.in_features, 100)
		self.model.load_state_dict(self.checkpoint['state_dict'])
		self.model.eval()

		self.criterion = CrossEntropyLoss()

		self.optimizer = Adam(self.model.parameters(), lr=1E-4)
		self.optimizer.load_state_dict(self.checkpoint['optimizer'])

	def classify(self, img):
		logps = self.model(img)

		return logps

