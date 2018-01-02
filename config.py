import argparse

def config():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_step",type=int,default=1000,help="train step")
    parser.add_argument("--model",type=str,default='autoencoder',help="type of autoencoder")
    parser.add_argument("--hidden_units",type=int,default=100,help='number of hidden units')
    parser.add_argument("--learning_rate",type=float,default=0.001,help='learn rate')
    parser.add_argument("--regularization",type=str,default='kl',help='if autoencoder is sparse autoencoder,it will decide which regularization should be used')
    parser.add_argument("--alpha",type=float,default=1.0,help='if the use l1 regularization,the alpha will decide the proportion of l1 loss')
    parser.add_argument("--sttdev",type=float,default=0.1,help="if autoencoder is denoise autoencoder,it will decide the sttdev of gaussian noise")

    return parser.parse_args()