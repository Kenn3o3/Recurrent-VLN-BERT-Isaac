import argparse
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--test_only', type=int, default=0, help='fast mode for testing')

        self.parser.add_argument('--iters', type=int, default=300000, help='training iterations')
        self.parser.add_argument('--name', type=str, default='default', help='experiment id')
        self.parser.add_argument('--vlnbert', type=str, default='oscar', help='oscar or prevalent')
        self.parser.add_argument('--train', type=str, default='listener')
        self.parser.add_argument('--description', type=str, default='no description\n')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=8)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')

        # Augmented Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.20)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--features", type=str, default='places365')

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Submision configuration
        self.parser.add_argument("--submit", type=int, default=0)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='adamW')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')
        # Simulation param

        self.parser.add_argument("--episode_index", default=0, type=int, help="Episode index.")
        self.parser.add_argument("--num_episodes", default=1, type=int, help="Number of episodes to run.")
        self.parser.add_argument("--task", type=str, default="go2_matterport", help="Name of the task.")
        self.parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
        self.parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
        self.parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
        self.parser.add_argument("--video_length", type=int, default=2000, help="Length of the recorded video (in steps).")
        self.parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
        self.parser.add_argument("--use_cnn", action="store_true", default=None, help="Name of the run folder to resume from.")
        self.parser.add_argument("--arm_fixed", action="store_true", default=False, help="Fix the robot's arms.")
        self.parser.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN in the actor-critic model.")
        self.parser.add_argument("--action_repeat", default=40, type=int, help="Number of simulation steps to repeat each action.")
        self.parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
        self.parser.add_argument("--vlnbert_model_path", type=str, required=False, help="Path to the VLNBert model checkpoint.")

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args

args.vlnbert = "prevalent"
args.iters = 10000
args.batchSize = 2
args.lr = 1e-5
args.name = "navigation_PREVALENT"
args.maxInput = 80
args.optim = 'adamW'
args.warmup_fraction = 0.1 