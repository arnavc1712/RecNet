import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
   

    # Model settings
    parser.add_argument(
        "--max_len",
        type=int,
        default=28,
        help='max length of captions(containing <sos>,<eos>)')

    parser.add_argument(
        '--dim_hidden',
        type=int,
        default=768,
        help='size of the rnn hidden layer')

    parser.add_argument(
        '--num_head',
        type=int,
        default=1,
        help='number of attention heads')

    parser.add_argument(
        '--include_user',
        type=int,
        default=0,
        help='Whether to include user embeddings or not')

    parser.add_argument(
        '--num_recs',
        type=int,
        default=10,
        help='number of recommendations to show')

    parser.add_argument(
        '--num_layers', type=int, default=2, help='number of layers in the Transformers')

    parser.add_argument(
        '--input_dropout_p',
        type=float,
        default=0.2,
        help='strength of dropout in the Pre Transformer Encoder Linear Layer')

    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=10,
        help='sequence length of the items')

    parser.add_argument(
        '--item_data_path',
        type=str,
        default="./data/ml-100k/u.item",
        help='data path for item info')

    parser.add_argument(
        '--seq_data_path',
        type=str,
        default="./sequences.pkl",
        help='data path for item sequences')


    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help="strength on dropout in Transformer Encoder layer")

    parser.add_argument(
        '--dim_item',
        type=int,
        default=50,
        help='the embedding size of each item'
    )
    parser.add_argument(
        '--dim_inner',
        type=int,
        default=100,
        help='Dimension of inner feature in Position wise feed forward neural network.')




    parser.add_argument(
        '--dim_model',
        type=int,
        default=50,
        help='dim of model')

    # Optimization: General

    parser.add_argument(
        '--epochs', type=int, default=6001, help='number of epochs')

    parser.add_argument(
        '--num_neg_sml', type=int, default=2, help='number of negative samples to consider for adaptive hinge loss')

    parser.add_argument(
        '--batch_size', type=int, default=128, help='minibatch size')

    parser.add_argument(
        '--grad_clip',
        type=float,
        default=5,  # 5.,
        help='clip gradients at this value')

    parser.add_argument(
        '--self_crit_after',
        type=int,
        default=-1,
        help='After what epoch do we start finetuning the CNN? \
                        (-1 = disable; never finetune, 0 = finetune from start)'
    )

    parser.add_argument(
        '--learning_rate', type=float, default=0.1, help='learning rate')

    parser.add_argument(
        '--learning_rate_decay_every',
        type=int,
        default=30,
        help='every how many iterations thereafter to drop LR?(in epoch)')

    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)

    parser.add_argument(
        '--optim_alpha', type=float, default=0.9, help='alpha for adam')

    parser.add_argument(
        '--optim_beta', type=float, default=0.999, help='beta used for adam')

    parser.add_argument(
        '--optim_epsilon',
        type=float,
        default=1e-8,
        help='epsilon that goes into denominator for smoothing')


    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4,
        help='weight_decay. strength of weight regularization')

    parser.add_argument(
        '--save_checkpoint_every',
        type=int,
        default=30,
        help='how often to save a model checkpoint (in epoch)?')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./save',
        help='directory to store check pointed models')

    parser.add_argument(
        '--loss',
        type=str,
        default='adaptive_hinge_loss',
        help='Loss function')

    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default='./save/bleu_38_epoch_240.pth',
        help='directory to load check pointed models')

    parser.add_argument(
        '--warm_up_steps',
        type=int,
        default=1000,
        help="number of warm up steps")

    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')

    args = parser.parse_args()

    return args
