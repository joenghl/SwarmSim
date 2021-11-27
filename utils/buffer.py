"""
Replay buffer for AirSim.
"""
import numpy as np
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer:
    """
    Replay buffer for multi-drone RL
    """
    def __init__(self, capacity, num_agents):
        self._storage = []
        self._maxsize = int(capacity)
        self._next_idx = 0
        self.num_agents = num_agents

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def push(self, obs, acs, rews, next_obs, dones):
        data = (obs, acs, rews, next_obs, dones)
        
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, to_gpu=False):
        """
        Output:
            (nagent, batch, space)
        """
        obs, acs, rews, next_obs, dones =\
            [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            ob, ac, rew, next_ob, done = data
            obs.append(np.array(ob, copy=False))
            acs.append(np.array(ac, copy=False))
            rews.append(rew)
            next_obs.append(np.array(next_ob, copy=False))
            dones.append(done)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)

        return ([cast(obs)[:, i, :] for i in range(self.num_agents)],
                [cast(acs)[:, i, :] for i in range(self.num_agents)],
                [cast(rews)[:] for i in range(self.num_agents)],
                [cast(next_obs)[:, i, :] for i in range(self.num_agents)],
                [cast(dones)[:] for i in range(self.num_agents)])
        
        


    def make_index(self, batch_size):
        return [np.random.randint(0, len(self._storage)) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 -i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size, to_gpu=False):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
