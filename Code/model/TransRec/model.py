import torch
import torch.nn as nn
import numpy as np
from model.TransRec.layers import multihead_attention,feedforward
from utils.utils import *


def get_mask(input_seq):
	return input_seq.ne(0).type(torch.float).unsqueeze(-1).cuda()

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask.cuda()

class Model(nn.Module):

	def __init__(self,num_users,num_items,opt):
		super().__init__()

		self.item_emb = nn.Embedding(num_embeddings=num_items+1,
									 embedding_dim=opt['dim_item'],
									 padding_idx=0)
									 # freeze=False) ## (batch_size, max_seq_len, dim_item)

		self.pos_emb = nn.Embedding(num_embeddings=opt['max_seq_len'],
									embedding_dim=opt['dim_item']
									)
									# padding_idx=0,
									# freeze=False)

		self.dropout = nn.Dropout(opt['dropout'])

		self.layer_norm = nn.LayerNorm(opt['dim_item'])

		self.multihead_attention = multihead_attention(num_units=opt['dim_item'],
													   num_heads=opt['num_head'],
													   dropout_rate=opt['dropout'])

		self.feedforward = feedforward(num_units=[opt['dim_item'],opt['dim_item']],
									   dropout_rate=opt['dropout'])

		self.num_users = num_users
		self.num_items = num_items
		self.opt = opt
		


	def get_user_rep(self,u,input_seq):
		mask = get_mask(input_seq)
		subseq_mask = get_subsequent_mask(input_seq)
		input_pos = pos_generate(input_seq).cuda()
		## Sequence embedding
		input_emb = self.item_emb(input_seq)

		## Positional encoding
		pos_emb = self.pos_emb(input_pos)

		input_emb = input_emb + pos_emb

		input_emb = self.dropout(input_emb)

		input_emb = input_emb * mask

		for i in range(self.opt['num_layers']):


			input_emb = self.multihead_attention(q=self.layer_norm(input_emb),
                                            	 k=input_emb,
                                            	 subseq_mask=subseq_mask
                                            	)

			input_emb = self.feedforward(self.layer_norm(input_emb))

			input_emb = input_emb * mask



			# input_emb = feedforward(self.layer_norm(input_emb),
			# 						num_units=[self.opt['dim_item'], self.opt['dim_item']],
   #                                  dropout_rate=self.opt['dropout'],
   #                                  )


		# 	input_emb = input_emb * mask

		input_emb = self.layer_norm(input_emb)

		return input_emb


	def forward(self,user_rep,item_seq):

		item_seq = item_seq.view(user_rep.shape[0]*self.opt['max_seq_len'])
		item_seq = self.item_emb(item_seq)
		user_rep = user_rep.view(user_rep.shape[0]*self.opt['max_seq_len'],self.opt['dim_item'])
		logits = torch.sum(item_seq*user_rep,-1)

		return logits

	def predict(self,user_rep,item_idx):
		# print(user_rep.squeeze().shape)
		
		item_emb = self.item_emb(item_idx)
		# print(item_emb.transpose(0,1).shape)
		test_logits = torch.matmul(user_rep.squeeze(),item_emb.transpose(0,1).contiguous())
		test_logits = test_logits.view(user_rep.shape[0],self.opt['max_seq_len'],101)
		return test_logits[:, -1, :]
		# test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
		# print(test_logits.shape)
		# print(user_rep.shape)







