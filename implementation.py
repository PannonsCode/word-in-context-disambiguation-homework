import numpy as np
import torch
import torch.nn
from typing import List, Tuple, Dict

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from typing import *
import string

from model import Model


def build_model(device: str) -> Model:
	# STUDENT: return StudentModel()
	# STUDENT: your model MUST be loaded on the device "device" indicates
	return StudentModel(device)

class RandomBaseline(Model):

	options = [
	('True', 40000),
	('False', 40000),
	]

	def __init__(self):

		self._options = [option[0] for option in self.options]
		self._weights = np.array([option[1] for option in self.options])
		self._weights = self._weights / self._weights.sum()

	def predict(self, sentence_pairs: List[Dict]) -> List[str]:
		return [str(np.random.choice(self._options, 1, p=self._weights)[0]) for x in sentence_pairs]
	


class StudentModel(Model):

	# STUDENT: construct here your model
	# this class should be loading your weights and vocabulary
	
	def __init__(self, dev: str):
		
		#load pre-trained classifier model
		self.model = torch.load('model/model.pt', map_location = dev)
		self.model.eval()
		
		#glove ready for use
		self.vocabulary = torch.load('model/vocabulary.txt')
		
		#variables used for elaboration
		self.SEP = 'SEP'
		self.TGT = 'TGT'
		self.dim_tensor = self.vocabulary['.'].shape[1]

	def predict(self, sentence_pairs: List[Dict]) -> List[str]:
		
		# STUDENT: implement here your predict function
		# remember to respect the same order of sentences!

		label = []
		
		for sp in sentence_pairs:
			
			#preparation of data
			x = self.organize_dataset(sp)
			x = self.VecFromWord(x)
			x = self.aggregation(x)

			#prediction and its elaboration
			out = self.model(x)
			pred = torch.round(out['pred'])
			pred = pred[0].item()
			
			if pred == 1:
				label.append('True')
			else:
				label.append('False')
				
		return label

	#initial organization of data
	def organize_dataset(self, line: Dict):

		#first sentence of dataset
		sentence1 = line['sentence1']
		sentence1 = sentence1[0:np.int(line['start1'])-1]+" "+self.TGT+sentence1[np.int(line['end1']):]
		sentence1 = ''.join( c for c in sentence1 if c not in string.punctuation) #remove punctuation from words

		#second sentence of dataset
		sentence2 = line['sentence2']
		sentence2 = sentence2[0:np.int(line['start2'])-1]+" "+self.TGT+sentence2[np.int(line['end2']):]
		sentence2 = ''.join( c for c in sentence2 if c not in string.punctuation) #remove punctuation from words

		#concatenate sentences with format "....<TGT>.....<SEP>.....<TGT>...."
		sentences = sentence1+" "+self.SEP+" "+sentence2
		
		return sentences


	#transformation of words in tensors using previos vocabulary
	def VecFromWord(self, sentences: str):
		
		'''for each sentence split the two different sentences of dataset
		   and tranform evry word in the corrispective tensors
		   given by pre-trained model'''

		split_id = sentences.index(self.SEP)
		s1 = sentences[:split_id]               #first sentence
		s2 = sentences[split_id+1:]				#second sentence
		vecs1 = self.transform(s1.split(" "))	#convert first sentence
		vecs2 = self.transform(s2.split(" "))	#converte second sentence

		new_sentence = []
		new_sentence.append(vecs1)
		new_sentence.append(vecs2)

		#format of output: [[tensor1_1,.....,tensor1_N],[tensor2_1,......,tensor2_N]]
		return new_sentence

	#function for effective transormation
	def transform(self, l: list):

		vec = []   #list for the word in vocabulary
		miss = []  #list for missing words

		for word in l:
			word = word.lower()
			if word in self.vocabulary.keys():
				vec.append(self.vocabulary[word])
			elif word != self.SEP and word != self.TGT:
				miss.append(word)

		'''management of miss words: each missing word will be replaced
		   with the average of the other words'''
		summ = 0
		for i in vec:
			summ += i
		summ = summ/len(vec)
		for _ in range(len(miss)):
			vec.append(summ)
		
		'''management of target words: they will be tensors with all zeros
		   added one time because there is one target word for each sentence'''
		vec.append(torch.zeros(self.dim_tensor))

		#return [tensor1,......,tensorN]
		return vec


	#final organization of data for input of the classification model
	def aggregation(self, data: list):
		
		new_data = 0

		#organize tensors of first sentence
		sum1 = 0
		for el in data[0]:
			sum1+=el
		sum1 = sum1/len(data[0])

		#organize tensors of second tensors
		sum2 = 0
		for el in data[1]:
			sum2+=el
		sum2 = sum2/len(data[1])

		#return a tensor with tensor_i = [tensor1_1,....,tensor1_N] + [tensor2_1,......,tensor2_N]
		new_data = sum1-sum2

		return new_data
		
#class of classifier model used
class SentenceClassifier(torch.nn.Module):
	
	def __init__(self, n_features):
		
		super().__init__()
		self.layerIn = torch.nn.Linear(n_features)
		self.hidden1 = torch.nn.Linear(100,50)
		self.hidden2 = torch.nn.Linear(50,10)
		self.layerOut = torch.nn.Linear(10,1)
		self.loss_fn = torch.nn.BCELoss()
		
	def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
		
		out = self.layerIn(x)
		out = torch.relu(out)
		out = self.hidden1(out)
		out = torch.relu(out)
		out = self.hidden2(out)
		out = torch.relu(out)
		out = self.layerOut(out)
		out = torch.sigmoid(out).squeeze(1)
		out = torch.round(out)
		
		result = {'pred': out}
		
		if y is not None:
			loss = self.loss_fn(out,y)
			result['loss'] = loss
		
		return result
		



























