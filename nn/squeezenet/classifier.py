import torch 
from torchvision.models import squeezenet1_0
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch import nn
import os

class Classifier:
	# INITIATE THE MODEL AND LOAD THE CHECKPOINT
	def __init__(self, current_folder = os.path.dirname(os.path.abspath(__file__))):
		self.device = torch.device("cpu")
		self.annot_folder = os.path.join(current_folder, 'classifier100.pt')
		self.checkpoint = torch.load(self.annot_folder, map_location='cpu')

		self.model = squeezenet1_0(pretrained=True)
		self.model.num_classes = 100
		self.model.classifier[1] = nn.Conv2d(512, 100, kernel_size=(1, 1), stride=(1, 1))
		self.model.load_state_dict(self.checkpoint['state_dict'])
		self.model.eval()

		self.criterion = CrossEntropyLoss()

		self.optimizer = SGD(self.model.parameters(), lr=1E-3, momentum=0.9) 
		self.optimizer.load_state_dict(self.checkpoint['optimizer'])

		self.trans = transforms.Compose([transforms.Resize((224, 224)),
																		transforms.ToTensor()
																		])

	def classify(self, img):
		logps = self.model(img)
		_, predicted = torch.max(logps, 1)

		return predicted

