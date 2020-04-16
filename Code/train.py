import torch
import torch.optim as optim
import opts
from torch.utils.data import DataLoader
from utils.utils import *
from data_loader import RecDataset
import torch.optim as optim
from losses import hinge_loss, adaptive_hinge_loss, binary_cross_entropy
import os

from model.TransRec.model import Model
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/recnet_transformer_exp1')



def train(loader,optimizer,model,opt,dataset):
	# model.train()
	# epoch_loss = 0.0
	# ix_to_item = loader.dataset.get_ix_to_item()
	# item_to_ix = loader.dataset.get_item_to_ix()
	model.train()
	for epoch in range(opt['epochs']):
		epoch_loss = 0.0
		epoch_pos_loss = 0.0
		epoch_neg_loss = 0.0
		iterations = 0
		for i,(user, seq, pos, neg,seq_len) in enumerate(loader):
			torch.cuda.synchronize()			
			optimizer.zero_grad()
			user = user.cuda()
			seq = seq.cuda()
			pos = pos.cuda()
			neg = neg.cuda()
			seq_len = seq_len.cuda()
			user_rep = model.get_user_rep(user,seq)

			pos = pos.view(seq.shape[0]*opt['max_seq_len'])
			neg = neg.view(seq.shape[0]*opt['max_seq_len'])
			pos_logits = model(user_rep,pos)
			neg_logits = model(user_rep,neg)

			istarget = pos.ne(0).type(torch.float).view(seq.shape[0]*opt['max_seq_len'])
			# print(pos)
			# print(istarget.shape)
			pos_loss = torch.sum((-torch.log(torch.sigmoid(pos_logits) + 1e-24)*istarget))
			neg_loss = torch.sum((-torch.log(torch.sigmoid(neg_logits) + 1e-24)*istarget))
			loss = torch.sum((-torch.log(torch.sigmoid(pos_logits) + 1e-24)*istarget) - (torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)*istarget))
			loss = loss/torch.sum(istarget)

			epoch_loss += loss.item()
			epoch_pos_loss += pos_loss.item()
			epoch_neg_loss += neg_loss.item()

			# print(sum(istarget))

			
			print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
			torch.cuda.synchronize()

			loss.backward()
			# torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1)
			optimizer.step()

			iterations += 1

		writer.add_scalar("Loss/overall_loss",epoch_loss/iterations,epoch)

		writer.add_scalar("Loss/positive_loss",epoch_pos_loss/iterations,epoch)

		writer.add_scalar("Loss/negative_loss",epoch_neg_loss/iterations,epoch)


		if epoch%10==0:
			t_test = evaluate(model.eval(), (dataset.user_train, dataset.user_valid, dataset.user_test, dataset.num_users, dataset.num_items), opt)
			t_valid = evaluate_valid(model.eval(), (dataset.user_train, dataset.user_valid, dataset.user_test, dataset.num_users, dataset.num_items), opt)
			

			print(f"Valid NDCG : {t_valid[0]}\t Valid HIT@10 : {t_valid[1]}")
			print(f"Test NDCG : {t_test[0]}\t Test HIT@10 : {t_test[1]}")

			writer.add_scalar("Evaluation/Validation/NDCG@10",t_valid[0],epoch)
			writer.add_scalar("Evaluation/Validation/HIT@10",t_valid[1],epoch)
			writer.add_scalar("Evaluation/Test/NDCG@10",t_test[0],epoch)
			writer.add_scalar("Evaluation/Test/HIT@10",t_test[1],epoch)

			model_path = os.path.join(opt['checkpoint_path'], f'model_transformer_{epoch}.pth')
			model_info_path = os.path.join(opt['checkpoint_path'], 'model_transformer_score.txt')
			torch.save(model.state_dict(), model_path)

			print('model saved to %s' % (model_path))
			with open(model_info_path, 'a') as f:
				f.write('model_%d, loss: %.6f\n' % (epoch, epoch_loss/iterations))



			# t_valid = evaluate_valid(model, dataset, args, sess)





def main(opt):
	dataset = RecDataset('train',opt,model="transformer")
	dataloader = DataLoader(dataset,batch_size=opt['batch_size'],shuffle=True)

	# model = Encoder(seq_len=opt['max_seq_len'],
 #            dim_item=opt["dim_item"],
 #            dim_user=opt["dim_item"],
 #            n_users=dataset.get_num_users(),
 #            n_items=dataset.get_num_items(),
 #            n_layers=opt["num_layer"],
 #            n_head=opt["num_head"],
 #            d_k=opt["dim_model"]//opt["num_head"],
 #            d_v=opt["dim_model"]//opt["num_head"],
 #            d_model=opt["dim_model"],
 #            d_inner=opt["dim_inner"],
 #            input_dropout_p=opt["input_dropout_p"],
 #            dropout=opt["dropout"])

	# model = model.cuda()

	model = Model(num_users=dataset.get_num_users(),
				  num_items=dataset.get_num_items(),
				  opt=opt)

	model.cuda()
	optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                                          betas=(0.9, 0.98), eps=1e-09,weight_decay=0.001)
	# optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
	# betas=(0.9, 0.98), eps=1e-09), opt["dim_model"], opt["warm_up_steps"])
	train(dataloader,optimizer,model,opt,dataset)


if __name__ == "__main__":
	print("Running")
	opt = opts.parse_opt()
	opt = vars(opt)
	main(opt)