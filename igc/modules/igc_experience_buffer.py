import random
import collections
import torch


class Buffer:
    """
    Buffer for the most recent experiences used in RL
    """
    def __init__(self, size, sample_size):
        """

        :param size: maximum size of the buffer.
        :param sample_size: size of the randomly sampled batch of experiences.
        """
        self._size = int(size)
        self._sample_size = sample_size
        self._buffer = collections.deque(maxlen=self._size)

    def add(self, state, action, reward, next_state, done=False):
        """
        Add an experience tuple to the buffer.

        :param state: state (tensor): a tensor array corresponding to the env state
        :param action: action (tensor): a tensor array corresponding to the action:
        :param reward: reward (tensor): tensor obtained from the env:
        :param next_state: next_state (tensor): tensor array corresponding to the next state:
        :param done: done (tensor/bool): the ``terminated`` flag for this transition —
            1.0 at a true MDP terminal (goal reached), so the target stops bootstrapping.
            Truncation (a step/time-limit cut) is NOT terminal and must stay 0.0.
        :return:
        """
        self._buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        """This method sample from the buffer, where buffer store batch for of experiences,
        and return a batch of experiences ``(state, action, reward, next_state, done)``.
        :return:
        """
        samples = self._buffer
        if len(self._buffer) >= self._sample_size:
            samples = random.sample(self._buffer, self._sample_size)

        state_batch = torch.stack([sample[0] for sample in samples], dim=0)
        action_batch = torch.stack([sample[1] for sample in samples], dim=0)
        reward_batch = torch.stack([sample[2] for sample in samples], dim=0)
        next_state_batch = torch.stack([sample[3] for sample in samples], dim=0)
        done_batch = self._stack_done(samples, reward_batch)

        state_batch = torch.reshape(state_batch, (-1, state_batch.size(-1)))
        action_batch = torch.reshape(action_batch, (-1, action_batch.size(-1)))
        reward_batch = torch.reshape(reward_batch, (-1,))
        next_state_batch = torch.reshape(next_state_batch, (-1, next_state_batch.size(-1)))
        done_batch = torch.reshape(done_batch, (-1,))

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def sample(self):
        """Randomly sample experiences from the replay buffer.
        Returns:
          (tuple): batch of experience (state, action, reward, next_state, done)
        """
        samples = self._buffer
        if len(self._buffer) >= self._sample_size:
            samples = random.sample(self._buffer, self._sample_size)

        state_batch = torch.stack([sample[0] for sample in samples], dim=0)
        action_batch = torch.stack([sample[1] for sample in samples], dim=0)
        reward_batch = torch.stack([sample[2] for sample in samples], dim=0)
        next_state_batch = torch.stack([sample[3] for sample in samples], dim=0)
        done_batch = self._stack_done(samples, reward_batch)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    @staticmethod
    def _stack_done(samples, reward_batch):
        """Stack the per-sample done flags into a tensor shaped like ``reward_batch``.

        Tolerates plain ``bool``/scalar done values (the ``add`` default) by
        broadcasting them to each sample's reward shape, so a terminal mask is
        always available even for experiences added without one.
        """
        done_items = []
        for sample, reward in zip(samples, reward_batch):
            done = sample[4] if len(sample) > 4 else False
            if not torch.is_tensor(done):
                done = torch.full_like(reward, float(bool(done)))
            done_items.append(done)
        return torch.stack(done_items, dim=0)
