import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.airsim_env import Env
from utils.buffer import ReplayBuffer
# from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

USE_CUDA = False  # torch.cuda.is_available()

def run(config):
    # model path
    model_dir = Path('./models') / config.env_id

    # model index
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # if not USE_CUDA:
    #     torch.set_num_threads(config.n_training_threads)
    env = Env()
    maddpg = MADDPG.init_from_env(tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents)
    time_limit = 60
    global_step = 0
    global_train_num = 0
    episode = 0
    while True:
        try:
            print("episode%i" % episode)
            time_step = 0
            episode += 1
            dones = False
            train_num = 0.0
            obs = env.reset()
            maddpg.prep_rollouts(device='cpu')
            explr_pct_remaining = max(0, config.n_exploration_eps - episode) / config.n_exploration_eps
            maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
            maddpg.reset_noise()
            episode_reward = 0

            while not dones and time_step < time_limit:
                time_step += 1
                global_step += 1
                torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False)
                            for i in range(maddpg.nagents)]
                # get actions as torch Variables
                torch_agent_actions = maddpg.step(torch_obs, explore=True)
                # convert actions to numpy arrays
                agent_actions = [ac.data.numpy().squeeze() for ac in torch_agent_actions]
                # rearrange actions to be per environment
                # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                next_obs, rewards, dones, infos = env.step(agent_actions)
                # print(agent_actions)
                episode_reward += rewards
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
                obs = next_obs


                if len(replay_buffer) >= config.batch_size and global_step >= config.train_rate:
                    print("training")
                    for _ in range(config.epoch):
                        if USE_CUDA:
                            maddpg.prep_training(device='gpu')
                        else:
                            maddpg.prep_training(device='cpu')

                        for a_i in range(maddpg.nagents):
                            sample = replay_buffer.sample(config.batch_size,
                                                            to_gpu=USE_CUDA)
                            maddpg.update(sample, a_i, logger=logger)
                        maddpg.update_all_targets()
                        maddpg.prep_rollouts(device='cpu')

            ep_rews = episode_reward / time_step
            print("ep_rews: ", ep_rews)
            logger.add_scalar('mean_episode_rewards', ep_rews, episode)

            if episode % config.save_interval == 0:
                os.makedirs(run_dir / 'incremental', exist_ok=True)
                maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (episode + 1)))
                maddpg.save(run_dir / 'model.pt')
                        
        except KeyboardInterrupt:
            env.disconnect()


    # for ep_i in range(0, config.n_episodes):
    #     print("Episodes %i of %i" % (ep_i + 1, config.n_episodes))
    #     obs = env.reset()
    #     # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
    #     maddpg.prep_rollouts(device='cpu')
    #     explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
    #     maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
    #     maddpg.reset_noise()
    #     episode_reward = 0
    #     for et_i in range(config.episode_length):
    #         # rearrange observations to be per agent, and convert to torch Variable
    #         torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False)
    #                      for i in range(maddpg.nagents)]
    #         # get actions as torch Variables
    #         torch_agent_actions = maddpg.step(torch_obs, explore=True)
    #         # convert actions to numpy arrays
    #         agent_actions = [ac.data.numpy().squeeze() for ac in torch_agent_actions]
    #         # rearrange actions to be per environment
    #         # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
    #         next_obs, rewards, dones, infos = env.step(agent_actions)
    #         print(agent_actions)
    #         episode_reward += rewards
    #         replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
    #         obs = next_obs
    #         t += 1
    #         if (len(replay_buffer) >= config.batch_size and
    #             (t % config.steps_per_update) == 0):
    #             if USE_CUDA:
    #                 maddpg.prep_training(device='gpu')
    #             else:
    #                 maddpg.prep_training(device='cpu')

    #             for a_i in range(maddpg.nagents):
    #                 sample = replay_buffer.sample(config.batch_size,
    #                                                 to_gpu=USE_CUDA)
    #                 maddpg.update(sample, a_i, logger=logger)
    #             maddpg.update_all_targets()
    #             maddpg.prep_rollouts(device='cpu')
    #     ep_rews = episode_reward / config.episode_length
    #     print("ep_rews: ", ep_rews)
    #     logger.add_scalar('mean_episode_rewards' % ep_rews, ep_i)

    #     if ep_i % config.save_interval == 0:
    #         os.makedirs(run_dir / 'incremental', exist_ok=True)
    #         maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
    #         maddpg.save(run_dir / 'model.pt')
        

    # maddpg.save(run_dir / 'model.pt')
    # env.close()
    # logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    # logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="drone")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    # parser.add_argument("--n_rollout_threads", default=1, type=int)
    # parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e4), type=int)
    parser.add_argument("--n_episodes", default=1e-6, type=int)
    parser.add_argument("--episode_length", default=50, type=int)
    parser.add_argument("--steps_per_update", default=10, type=int)
    parser.add_argument('--epoch',      type=int,   default=1)
    parser.add_argument("--train_rate", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=512, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()

    run(config)
