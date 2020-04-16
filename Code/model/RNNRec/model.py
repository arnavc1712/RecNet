import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, num_users,num_items,opt,rnn_cell='gru'):
        """

        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(Model, self).__init__()
        self.dim_item = opt['dim_item']
        self.opt = opt
        self.dropout = opt['dropout']
        self.rnn_cell = rnn_cell

        self.item_emb = nn.Embedding(num_embeddings=num_items+1,
                                     embedding_dim=opt['dim_item'],
                                     padding_idx=0)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        # self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
        #                         bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self.rnn = self.rnn_cell(self.dim_item, self.dim_item, opt['num_layers'], batch_first=True,
                                 dropout=self.dropout)


    def get_user_rep(self,u,input_seq,seq_len):
        '''
        u: batch_size * 1
        input_seq: batch_size x max_len
        '''
        item_feats = self.item_emb(input_seq)
        sorted_values,sorted_indices = seq_len.sort(0, descending=True)
        pack = torch.nn.utils.rnn.pack_padded_sequence(item_feats[sorted_indices], batch_first=True, lengths=sorted_values)
        self.rnn.flatten_parameters()

        state1=None
        output, hidden = self.rnn(item_feats,state1)

        return output


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
