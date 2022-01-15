import math
import numpy as np
import torch
import matplotlib.pyplot as plt

class Evaluation:

	# Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...
	def compute_rank1(self, Y, y, classes):
		total = 0
		sum = 0
		for el in y:
			_, predicted = torch.max(el, 1)
			pred = classes[predicted.item()-1]
			if pred == Y[total]:
				sum +=1

			total += 1
		return round(sum/total*100, 2)

	def compute_rank5(self, Y, y, classes):
		total = 0
		sum = 0
		for el in y:
			_, top5 = torch.topk(el, 5)

			pred = []
			nparray = top5.numpy()[0]
			for e in nparray:
				pred.append(classes[e-1])
			
			if Y[total] in pred:
				sum +=1
			total += 1
		return round(sum/total*100, 2)
	
	def plot_histogram(selr, arr, title, value, labels):
		x = np.arange(len(arr))
		plt.bar(x, height=arr)
		#plt.bar(x_pos, height, color = (0.5,0.1,0.5,0.6))
		plt.xticks(x, labels)
		plt.ylim(0, 100)
		plt.title(title)
		plt.ylabel(value)
		plt.show()