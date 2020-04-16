import os
import json
import torch
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from utils.utils import *


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class RecDataset(Dataset):


	def get_num_items(self):
		return self.num_items

	def get_num_users(self):
		return self.num_users

	def __init__(self,mode,opt,model="transformer"):
		super(RecDataset,self).__init__()

		self.mode = mode
		self.model = model
		self.maxlen = opt['max_seq_len']

		
		[self.user_train,self.user_valid,self.user_test,self.num_users,self.num_items] = data_partition("ml-1m.txt")
		
		print(f"Total number of examples is {len(self.user_train)}")
		

	def __getitem__(self,ix):
		
		user = ix+1
		# print(sequence.shape)
		seq = np.zeros([self.maxlen], dtype=np.int32)
		pos = np.zeros([self.maxlen], dtype=np.int32)
		neg = np.zeros([self.maxlen], dtype=np.int32)
		
		
		ts = set(self.user_train[user])
		seq_len = min(len(self.user_train[user]),self.maxlen)

		if self.model=="transformer":
			idx = self.maxlen - 1
			nxt = self.user_train[user][-1]
			for i in reversed(self.user_train[user][:-1]):
				seq[idx] = i
				pos[idx] = nxt
				if nxt != 0: neg[idx] = random_neq(1, self.num_items + 1, ts)
				nxt = i
				idx -= 1
				if idx == -1: break
		else:
			idx = 0
			i = self.user_train[user][0]
			for nxt in self.user_train[user][1:]:
				seq[idx] = i
				pos[idx] = nxt
				if nxt != 0: neg[idx] = random_neq(1, self.num_items + 1, ts)
				i = nxt
				idx += 1
				if idx == self.maxlen: break

		seq = torch.from_numpy(seq).type(torch.LongTensor)
		pos = torch.from_numpy(pos).type(torch.LongTensor)
		neg = torch.from_numpy(neg).type(torch.LongTensor)
		return (user, seq, pos, neg,torch.tensor(seq_len))
	


	def __len__(self):
		return self.num_users







