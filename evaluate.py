import argparse
import torch
import time
import numpy as np
from torch.autograd import Variable
from pathlib import Path
from utils.airsim_env import Env
from algorithms.maddpg import MADDPG

# TODO: normalization


def run(config):
    model_path = (Path('./models') / config.env_id /
                  ('run%i' % config.run_num))
    
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                    config.incremental)


    maddpg = MADDPG.init_from_save(model_path)
    env = Env()
    maddpg.prep_rollouts(device="cpu")
    time_limit = 500
    episode = 0
    while True:
        try:
            print("episode %i begin" % (episode))
            episode += 1
            time_step = 0
            dones = False
            obs = env.reset()
            while not dones and time_step < time_limit:
                time_step += 1
                torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False)
                            for i in range(maddpg.nagents)]
                # print(torch_obs[0])
                torch_agent_actions = maddpg.step(torch_obs, explore=False)
                # convert actions to numpy arrays
                agent_actions = [ac.data.numpy().squeeze() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                # print(agent_actions)
                # agent_actions = [np.array([0.,0.,0.,1.,0.]), np.array([0.,0.,0.,1.,0.]),np.array([0.,0.,0.,1.,0.])]
                obs, rewards, dones, infos = env.step(agent_actions)  
                print(rewards)           

        except KeyboardInterrupt:
            env.disconnect()
            break
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_spread")
    parser.add_argument("--model_name", default="123",
                        help="123")
    parser.add_argument("--run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=24001, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)
