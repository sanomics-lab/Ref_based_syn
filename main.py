import os
import time
import random
import torch 
import argparse
from env import SynthesisEnv
from td3 import TD3
from dock import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=2)
parser.add_argument('--max_action', type=int, default=2000, help='max step to generate molecule')
parser.add_argument('--ckpt_dir', type=str, default='ckpt/', help='the dir for saving models')
parser.add_argument('--predictor', type=str, default='vina', help='which reward predictor to be used')

parser.add_argument('--state_dim', type=int, default=256)
parser.add_argument('--t_dim', type=int, default=300, help='the number of used templates')
parser.add_argument('--act_dim', type=int, default=256)

parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--action_noise', type=float, default=0.1)
parser.add_argument('--policy_noise', type=float, default=0.2)
parser.add_argument('--policy_noise_clip', type=float, default=0.2)
parser.add_argument('--temp', type=float, default=1.0, help='the temperature for gumbel softmax')

parser.add_argument('--delay_time', type=int, default=2)
parser.add_argument('--max_size', type=int, default=1e6)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

model_dir = os.path.join(args.ckpt_dir, 'TD3_model')
target_model_dir = os.path.join(args.ckpt_dir, 'TD3_target_model')
create_dir(model_dir)
create_dir(target_model_dir)
timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())

docking_config = get_docking_config_for_vina()
predictor = DockingVina(docking_config)

env = SynthesisEnv()
with open('data/matched_bbs.txt', 'r') as f:
    starting_smi = [l.strip() for l in f.readlines()]
    print(len(starting_smi))
with open('data/rxn_set_f.txt', 'r') as f:
    rxns = [l for l in f.readlines()]
    args.t_dim = len(rxns)
    
env.init(args, start_smis=starting_smi, predictor=predictor, max_action=args.max_action)
td3 = TD3(env, state_dim=args.state_dim, t_dim=args.t_dim, action_dim=args.act_dim, 
            ckpt_dir=args.ckpt_dir, gamma=args.gamma, tau=args.tau, action_noise=args.action_noise, 
            policy_noise=args.policy_noise, policy_noise_clip=args.policy_noise_clip,
            delay_time=args.delay_time, max_size=int(args.max_size), batch_size=args.batch_size,
            max_episodes=args.max_episodes, max_action=args.max_action, temperature=args.temp)
td3.train(timestamp)
env.close()
