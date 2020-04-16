import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import defaultdict
import os
import sys
import copy



# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        # self.loss_fn = nn.NLLLoss(reduce=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = target.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct


def show_predictions(input_ids,target_ids,user_rep,model,ix_to_item,attns,opt):
    random_id = random.randint(0, len(input_ids)-1)

    print("Sequence")
    target_ids=target_ids.cpu()
    input_ids = input_ids.cpu()
    # user_rep = user_rep.cpu()
    # model = model.cpu()
    print(list(map(lambda x:"UNK" if x not in ix_to_item else ix_to_item[x],input_ids[random_id].numpy().flatten())))
    print("\n")
    target = target_ids[random_id][-1:]
    user_rep = user_rep[random_id]

    item_ids = np.array(list(ix_to_item.keys())).reshape(-1,1)
    item_ids = torch.from_numpy(item_ids).type(torch.LongTensor).cuda()
    size = (len(item_ids),) + user_rep.size()
    out = model(user_rep.expand(*size),item_ids)
    preds = out.detach().cpu().numpy().flatten()

    most_probable_10 = preds.argsort()[-opt["num_recs"]:][::-1]
    most_prob_10_items = list(map(lambda x:"UNK" if x not in ix_to_item else ix_to_item[x],most_probable_10))
    g_t = ix_to_item[target.detach().numpy().flatten()[0]]

    print("Most probable")
    print(most_prob_10_items)
    print("\n")

    print("Attnentions")
    print(attns[random_id][-1])

    print("True Label")
    print(g_t)
    print("\n")




def pos_generate(item_seq):

    seq = list(range(item_seq.shape[1]))
    src_pos = torch.tensor([seq] * item_seq.shape[0])

    return src_pos


def pos_emb_generation(visual_feats, word_labels):
    '''
        Generate the position embedding input for Transformers.
    '''
    seq = list(range(1, visual_feats.shape[1] + 1))
    src_pos = torch.tensor([seq] * visual_feats.shape[0]).cuda()

    seq = list(range(1, word_labels.shape[1] + 1))
    tgt_pos = torch.tensor([seq] * word_labels.shape[0]).cuda()
    binary_mask = (word_labels != 0).long()

    return src_pos, tgt_pos*binary_mask

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(os.path.join(os.getcwd(),"Data",fname), 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def evaluate(model, dataset, opt):
    model.eval()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([opt['max_seq_len']], dtype=np.int32)
        idx = opt['max_seq_len'] - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        user_rep = model.get_user_rep(torch.tensor([u]).cuda(),torch.from_numpy(seq.reshape(1,-1)).type(torch.LongTensor).cuda())
        predictions = -model.predict(user_rep.cuda(),torch.tensor(item_idx).cuda())
        # predictions = -model.predict(torch.tensor([u]), torch.from_numpy(seq).type(torch.LongTensor), torch.tensor(item_idx))
        predictions = predictions.detach().cpu()[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.'),
        #     sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def evaluate_valid(model, dataset, opt):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([opt['max_seq_len']], dtype=np.int32)
        idx = opt['max_seq_len'] - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        user_rep = model.get_user_rep(torch.tensor([u]).cuda(),torch.from_numpy(seq.reshape(1,-1)).type(torch.LongTensor).cuda())
        predictions = -model.predict(user_rep.cuda(),torch.tensor(item_idx).cuda())
        predictions = predictions.detach().cpu()[0]
        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user

def evaluateRNN(model, dataset, opt):
    model.eval()
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([opt['max_seq_len']], dtype=np.int32)
        seq_len = min(len(train[u]) + 1,opt['max_seq_len'])
        idx = seq_len - 1
        
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        user_rep = model.get_user_rep(torch.tensor([u]).cuda(),torch.from_numpy(seq.reshape(1,-1)).type(torch.LongTensor).cuda(),torch.tensor([seq_len]).cuda())
        predictions = -model.predict(user_rep.cuda(),torch.tensor(item_idx).cuda())
        # predictions = -model.predict(torch.tensor([u]), torch.from_numpy(seq).type(torch.LongTensor), torch.tensor(item_idx))
        predictions = predictions.detach().cpu()[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.'),
        #     sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def evaluateRNN_valid(model, dataset, opt):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([opt['max_seq_len']], dtype=np.int32)
        seq_len = min(len(train[u]) + 1,opt['max_seq_len'])
        idx = seq_len - 1
        
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        user_rep = model.get_user_rep(torch.tensor([u]).cuda(),torch.from_numpy(seq.reshape(1,-1)).type(torch.LongTensor).cuda(),torch.tensor([seq_len]).cuda())
        predictions = -model.predict(user_rep.cuda(),torch.tensor(item_idx).cuda())
        # predictions = -model.predict(torch.tensor([u]), torch.from_numpy(seq).type(torch.LongTensor), torch.tensor(item_idx))
        predictions = predictions.detach().cpu()[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.'),
        #     sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# def show_prediction(seq_probs, labels, vocab):
#     '''
#         :return: predicted words and GT words.
#     '''
#     # Print out the predicted sentences and GT
#     _ = seq_probs.view(labels.shape[0], labels[:, :-1].shape[1], -1)[0]
#     pred_idx = torch.argmax(_, 1)
#     print(' \n')
#     print([vocab[str(widx.cpu().numpy())] for widx in pred_idx if widx != 0])
#     print([vocab[str(word.cpu().numpy())] for word in labels[0] if word != 0])

