import torch

import torch.nn.functional as F

def hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Hinge pairwise loss function.
    Parameters
    ----------
    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    """

    loss = torch.clamp(negative_predictions -
                       positive_predictions +
                       1.0, 0.0)

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()


def adaptive_hinge_loss(positive_predictions, negative_predictions, mask=None):
    """
    Adaptive hinge pairwise loss function. Takes a set of predictions
    for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closes to violating the
    ranking implicit in the pattern of user interactions.
    Approximates the idea of weighted approximate-rank pairwise loss
    introduced in [2]_
    Parameters
    ----------
    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Iterable of tensors containing predictions for sampled negative items.
        More tensors increase the likelihood of finding ranking-violating
        pairs, but risk overfitting.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    References
    ----------
    .. [2] Weston, Jason, Samy Bengio, and Nicolas Usunier. "Wsabie:
       Scaling up to large vocabulary image annotation." IJCAI.
       Vol. 11. 2011.
    """

    highest_negative_predictions, _ = torch.max(negative_predictions, 0)

    return hinge_loss(positive_predictions, highest_negative_predictions.squeeze(), mask=mask)


def binary_cross_entropy(positive_predictions,negative_predictions,mask):
    '''
        Implementing Binary Cross entropy loss

        positive_predictions (batch_size,max_seq_len)
        negative_predictions (n,batch_size,max_seq_len)
    '''
    
    positive_predictions = torch.sigmoid(positive_predictions)
    negative_predictions = torch.sigmoid(negative_predictions)


    log_pos = torch.log(positive_predictions)
    log_neg = torch.log(1.0-negative_predictions)
    tot_neg_predictions = torch.sum(log_neg,0).squeeze() ## (batch_size,max_seq_len)
    
    loss = log_pos + log_neg
    

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return -(loss.sum() / mask.sum())

    return -loss.mean()
