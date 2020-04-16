import torch
import torch.nn as nn
import numpy as np
from utils.utils import *



class multihead_attention(nn.Module):

	def __init__(self,num_units,num_heads,dropout_rate):
		super().__init__()
		self.d_q = num_units//num_heads
		self.num_units = num_units
		self.num_heads = num_heads

		self.Q = nn.Linear(num_units, num_heads * self.d_q)
		self.K = nn.Linear(num_units, num_heads * self.d_q)
		self.V = nn.Linear(num_units, num_heads * self.d_q)

		self.softmax = nn.Softmax(dim=2)
		self.dropout = nn.Dropout(dropout_rate)


	def forward(self,q,k,subseq_mask):
		d_q, n_head = self.d_q,self.num_heads
		residual = q

		sz_b, len_q, _ = q.size()
		sz_b, len_k, _ = k.size()
		sz_b, len_v, _ = k.size()

		q = self.Q(q).view(sz_b, len_q, n_head, d_q)
		k = self.K(k).view(sz_b, len_k, n_head, d_q)
		v = self.V(k).view(sz_b, len_v, n_head, d_q)

		q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_q) # (n*b) x lq x dk
		k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_q) # (n*b) x lk x dk
		v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_q) # (n*b) x lv x dv

		outputs = torch.bmm(q, k.transpose(1, 2))
		outputs = outputs / np.power(d_q, 0.5)

		key_masks = torch.sign(torch.abs(torch.sum(k,-1)))
		key_masks = key_masks.repeat(n_head,1)
		key_masks = torch.unsqueeze(key_masks, 1).repeat(1, q.shape[1], 1)


		outputs = outputs.masked_fill(key_masks.eq(0),(-2**32+1))

		mask = subseq_mask
		mask = mask.repeat(n_head,1,1)

		outputs = outputs.masked_fill(mask.eq(1),(-2**32+1))

		outputs = self.softmax(outputs)


		query_masks = torch.sign(torch.abs(torch.sum(q,-1)))
		query_masks = query_masks.repeat(n_head,1)
		query_masks = torch.unsqueeze(query_masks, -1).repeat(1,1,k.shape[1])

		outputs = outputs * query_masks

		outputs = self.dropout(outputs)

		outputs = torch.bmm(outputs, v)

		outputs = outputs.view(n_head, sz_b, len_q, d_q)
		outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

		outputs = outputs + residual

		return outputs








class feedforward(nn.Module):
	def __init__(self, num_units,dropout_rate):
		super().__init__()
		self.w_1 = nn.Conv1d(num_units[0], num_units[1], 1) # position-wise
		self.w_2 = nn.Conv1d(num_units[1], num_units[0], 1) # position-wise
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, x):
		residual = x
		# x = self.layer_norm(x)
		output = x.transpose(1, 2)
		output = self.w_1(output)
		output = self.dropout(output.transpose(1,2))
		output = output.transpose(1,2)
		output = self.w_2(output)
		output = output.transpose(1, 2)
		output = self.dropout(output)
		# output = self.layer_norm(output + residual)
		output = output + residual
		return output

