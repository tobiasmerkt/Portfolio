import numpy as np
from itertools import count
from collections import deque
from collections import namedtuple
from typing import List
import shutil  # for copying models
import pickle
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
import gymnasium
import Environments
from helper_functions import print_checks, save_files, load_models

torch.manual_seed(1)

# tensorboard --logdir=History/Tensorboard


class Fight:
    """
    RL part of the ancient roman fight simulation. Fight creates an instance of the
    environment and uses RL to learn how to position the individual fighters for best
    outcomes. Currently, it uses a simple actor-critic infrastructure for quick,
    iterative development but PPO implementation is planned.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        episode_length: int = 400,
        episode_count: int = 10000,
        max_steps: int = 400,
        size: int = 25,
        episodes_per_model: int = 25,
        lr_schedule_step_size: int = 50,
        lr_schedule_gamma: float = 0.90,
        entropy_loss_factor: float = 5e4,
        clip_grad_norm: float = 30.0,
        team0: List[int] = [10, 5, 10],  # swords, archers, cavalry
        team1: List[int] = [14, 7, 10],
        log_interval: int = 1,
    ):
        """
        Constructor for a Fight model.

        Parameters
        ----------
        lr : float
                Learning rate
        episode_length : int
                How many simulation steps are performed between backpropagations.
        max_steps : int
                Maximum number of environment steps. If reached, the env resets.
        size : int
                Side length of the squared battle field in metres.
        episodes_per_model : int
                After how many episodes of self-play, the enemy will be updated.
        lr_schedule_step_size: int
                Episode count until learning rate is multiplied with lr_schedule_gamma.
        lr_schedule_gamma : float
                Learning rate scheduler multiplication rate.
        entropy_loss_factor : float
                Number by which entropy loss will be divided. Was optimised using bayes
                optimisation and should only be changed if necessary.
        clip_grad_norm : float
                Gradients will be clipped to this number.
        team_0/1 : List[int,int,int]
                How many fighters per fighter type exist for each team. Refers to number
                of [sword fighters, pikes, cavalry]
        log_interval : int
                After how many episodes the current progress will be printed.
        """
        self.lr = lr
        self.episode_length = int(episode_length)
        self.episode_count = int(episode_count)
        self.max_steps = int(max_steps)
        self.size = int(size)
        self.episodes_per_model = int(episodes_per_model)
        self.lr_schedule_step_size = lr_schedule_step_size
        self.lr_schedule_gamma = lr_schedule_gamma
        self.entropy_loss_factor = entropy_loss_factor
        self.clip_grad_norm = clip_grad_norm
        self.swords_team0 = team0[0]
        self.archers_team0 = team0[1]
        self.cavalry_team0 = team0[2]
        self.swords_team1 = team1[0]
        self.archers_team1 = team1[1]
        self.cavalry_team1 = team1[2]
        self.n_fighters_t0 = self.swords_team0 + self.archers_team0 + self.cavalry_team0
        self.n_fighters_t1 = self.swords_team1 + self.archers_team1 + self.cavalry_team1
        self.n_fighters = self.n_fighters_t0 + self.n_fighters_t1
        self.log_interval = log_interval
        self.writer = SummaryWriter(log_dir="../MARL/History/Tensorboard")
        # initial_x = np.array(
        #     [0.40 + 0.05 * np.random.rand() for _ in range(self.n_fighters_t0)]
        #     + [0.60 + 0.05 * np.random.rand() for _ in range(self.n_fighters_t1)]
        # )
        # initial_y = np.random.rand(self.n_fighters)
        self.reset_options = None  # {"initial_x": initial_x, "initial_y": initial_y}
        self.create_env()  # creates MARL environment
        self.create_AC()  # creates Actor, Critic, Optimizer

    def create_env(self):
        """
        Creates an instance of the custom MARL environment which outputs data for each
        fighter individually -> output is array of shape (m fighters,8) where 8 refers
        to the observation of each fighter.
        """

        env = gymnasium.make(
            "MarlEnv-v0",
            size=self.size,
            max_steps=self.max_steps,
            swords_team0=self.swords_team0,
            archers_team0=self.archers_team0,
            cavalry_team0=self.cavalry_team0,
            swords_team1=self.swords_team1,
            archers_team1=self.archers_team1,
            cavalry_team1=self.cavalry_team1,
            disable_env_checker=True,  # suppress reward type warning
        )
        self.env = env

    def create_AC(self):
        """
        Creates the actor and critic and their respective vmap method to let them run
        for all fighters simultaneously. It also creates an optimizer with a StepLR
        scheduler.
        """

        class ActorNet(nn.Module):
            def __init__(self, input: int):
                super(ActorNet, self).__init__()
                self.linear_tanh = nn.Sequential(
                    nn.Linear(input, 16),
                    nn.Tanh(),
                    nn.Linear(16, 16),
                    nn.Tanh(),
                )
                self.last_layer = nn.Linear(16, 5)

                self.saved_actions = []
                self.rewards = []
                self.entropy_losses = []

            def forward(self, x):
                x = self.linear_tanh(x)
                x = self.last_layer(x)
                return F.softmax(x, dim=-1)

        class CriticNet(nn.Module):
            def __init__(self, input: int):
                super(CriticNet, self).__init__()
                self.linear_tanh = nn.Sequential(
                    nn.Linear(input, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1),
                )

                self.saved_actions = []

            def forward(self, x):
                x = self.linear_tanh(x)
                return x

        actor = ActorNet(input=8)
        # Reduce last policy layer weights
        actor.last_layer.weight.data = actor.last_layer.weight.data / 100
        critic = CriticNet(input=8)
        self.enemy = copy.deepcopy(actor)
        self.enemy.eval()
        params_list = list(actor.parameters()) + list(critic.parameters())
        optimizer = optim.Adam(params_list, lr=self.lr)
        scheduler = StepLR(
            optimizer,
            step_size=self.lr_schedule_step_size,
            gamma=self.lr_schedule_gamma,
        )
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eps = np.finfo(np.float32).eps.item()
        self.apply_actor = torch.vmap(self.actor)
        self.apply_critic = torch.vmap(self.critic)
        self.apply_enemy = torch.vmap(self.enemy)

    def select_action(self, state, alive_fighter):
        """
        Selects the actions for all fighters. Fighters can move right, up, left, down,
        or stay at their current position.

        Parameters
        ----------
        state : np.ndarray
                Array with the environment state and shape (n fighters,8). 6 of the 8
                variables refer to the potential field to all 6 friendly and enemy
                fighter types. The remaining variables are the fighter team and type.
        alive_fighter : dict
                Dead fighters are removed from the simulation. Since self-play is used
                and the actor of team 1 will be updated every self.episodes_per_model
                steps, this dict provides information how many fighters per team are
                still alive. This number is then used to select the right model for each
                team.

        Returns
        -------
        full_actions : np.ndarray
                Numpy array with actions for each fighter and shape (n_fighters,). Each
                entry is an integer corresponding to the 5 possible actions.
        m_t0.probs.std().item(): float
                Standard deviation of the action probability distribution. Used for
                debugging.
        """
        alive_t0 = alive_fighter["alive_t0"]
        state_t0 = torch.from_numpy(state[:alive_t0, :]).float().unsqueeze(0)
        probs_t0 = self.apply_actor(state_t0)
        probs_t0 = torch.squeeze(probs_t0)  # remove dim with size 1
        m_t0 = Categorical(probs=probs_t0)
        actions_t0 = m_t0.sample()
        entropy_loss = torch.sum(torch.multiply(probs_t0, torch.log(probs_t0)))
        self.actor.saved_actions.append(m_t0.log_prob(actions_t0))
        self.actor.entropy_losses.append(entropy_loss)
        state_values = self.apply_critic(state_t0)
        self.critic.saved_actions.append(state_values.squeeze())

        state_t1 = torch.from_numpy(state[alive_t0:, :]).float().unsqueeze(0)
        probs_t1 = self.apply_enemy(state_t1)
        probs_t1 = torch.squeeze(probs_t1)  # remove dim with size 1
        m_t1 = Categorical(probs=probs_t1)
        actions_t1 = m_t1.sample()

        actions_t0_np = actions_t0.numpy()
        actions_t1_np = actions_t1.numpy()
        full_actions = np.concatenate([actions_t0_np, actions_t1_np], axis=None)

        return full_actions, m_t0.probs.std().item()

    def finish_episode(self, episode, idx_alive_t0):
        """
        Performs the RL part at the end of each episode. Uses simple Actor-Critic to
        find a good policy.

        Parameters
        ----------
        episode : int
                Episode number used for the Tensorboard writer.
        idx_alive_t0 : List[int]
                Indices of alive fighters of team 0. Necessary to track reward of each
                fighter because dead fighters are removed from simulation.

        Returns
        -------
        loss.detach().numpy() : np.ndarray
                Episode loss used to display the running loss and historically for
                parameter tuning with Bayes Optimisation.
        """
        R = 0
        returns = []
        # calculate returns
        for r in self.actor.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(np.array(returns))

        # Standardise returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # To keep dimensionality constant, fill up dead fighters with zero
        full_values = torch.zeros(
            size=(len(self.critic.saved_actions), self.n_fighters_t0),
            dtype=torch.float32,
        )
        full_logprobs = torch.zeros(
            size=(len(self.actor.saved_actions), self.n_fighters_t0),
            dtype=torch.float32,
        )
        for i in range(len(self.critic.saved_actions)):
            mask = torch.tensor(idx_alive_t0[i], dtype=int)
            full_values[i, mask] = self.critic.saved_actions[i]
            full_logprobs[i, mask] = self.actor.saved_actions[i]

        advantage = returns - full_values
        policy_loss = (-full_logprobs * advantage).sum()
        apply_huber_loss = torch.vmap(F.huber_loss)
        value_loss = apply_huber_loss(full_values, returns).sum()
        entropy_loss = torch.sum(torch.stack(self.actor.entropy_losses))
        self.optimizer.zero_grad()
        loss = value_loss + policy_loss + entropy_loss / self.entropy_loss_factor
        loss.backward()
        # Clip gradients for greater stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.writer.add_scalar("Loss/Actor", policy_loss, global_step=episode)
        self.writer.add_scalar("Loss/Critic", value_loss, global_step=episode)
        self.writer.add_scalar(
            "Loss/Entropy",
            entropy_loss / self.entropy_loss_factor,
            global_step=episode,
        )

        # reset rewards and action buffer
        del self.actor.rewards[:]
        del self.actor.saved_actions[:]
        del self.actor.entropy_losses[:]
        del self.critic.saved_actions[:]

        return loss.detach().numpy()

    def run_episode(self, episode):
        """
        Performs all steps of a single episode -> lets environment run for fixed number
        of steps and uses data to learn RL afterwards. It then writes statistics with
        the Tensorboard Writer.

        Parameters
        ----------
        episode : int
                Episode number used for the Tensorboard writer.

        Returns
        -------
        ep_loss : float
                Total loss at this episode.
        ep_reward : float
                Sum of all fighter rewards of team 0.
        """
        ep_reward = 0
        ep_prob_std = []
        ep_actions = []
        ep_idx_alive_t0 = []
        state, info = self.env.reset(
            seed=np.random.randint(low=1),
            options=self.reset_options,
        )
        ep_idx_alive_t0.append(info["idx_alive_t0"])
        for _ in range(1, self.episode_length + 1):
            action, std = self.select_action(state, info)
            state, reward, done, _, info = self.env.step(action)
            self.actor.rewards.append(reward)
            ep_reward += np.sum(reward)
            ep_prob_std.append(std)
            ep_actions.append(action)
            ep_idx_alive_t0.append(info["idx_alive_t0"])
            if done:
                state, _ = self.env.reset(
                    seed=np.random.randint(low=1),
                    options=self.reset_options,
                )
        ep_loss = self.finish_episode(episode, ep_idx_alive_t0)
        self.writer.add_scalar(
            "Prob std", np.mean(np.array(ep_prob_std)), global_step=episode
        )
        self.writer.add_scalar(
            "Episode reward", np.mean(np.array(ep_reward)), global_step=episode
        )
        self.writer.add_histogram(
            "Actions", np.concatenate(ep_actions, axis=None), global_step=episode
        )

        return ep_loss, ep_reward

    def train_model(self, print=False):
        """
        Performs all steps necessary to train the model. This method will e.g. be called
        by the multithreading module.

        Parameters
        ----------
        print : bool
                Whether the learning progress will be printed or not. Usually turned on
                when running single task or when debugging and turned off when using
                multithreading/ genetic algorithms.
        """
        running_loss = deque(maxlen=100)
        running_reward = deque(maxlen=100)
        for i_episode in range(self.episode_count):
            ep_loss, ep_reward = self.run_episode(i_episode)
            running_loss.appendleft(ep_loss)
            running_reward.appendleft(ep_reward)

            if i_episode % self.episodes_per_model == 0:
                self.enemy = copy.deepcopy(self.actor)
                self.enemy.eval()
            if print:
                print_checks(
                    i_episode,
                    ep_reward,
                    running_reward,
                    running_loss,
                    save_files=False,
                    log_interval=self.log_interval,
                    writer=self.writer,
                    episode_count=self.episode_count,
                )
        if print:
            return np.mean(running_loss)

    def evaluate_model(self):
        """
        After RL training has finished, the new model is tested against the old model.

        Returns
        -------
        reward : float
                Mean reward of the new model vs the old model.
        """
        path = "../MARL/History/Models/Temp_models/best_model.pth"
        checkpoint = torch.load(path)
        self.enemy.load_state_dict(checkpoint["actor"])
        self.enemy.eval()
        rew_arr = []
        state, alive_fighters = self.env.reset(
            seed=np.random.randint(low=1),
            options=self.reset_options,
        )
        for _ in range(8000):
            action, std = self.select_action(state, alive_fighters)
            state, reward, done, _, alive_fighters = self.env.step(action)
            rew_arr.append(np.sum(reward))

            if done:
                state, alive_fighters = self.env.reset(
                    seed=np.random.randint(low=1),
                    options=self.reset_options,
                )
        return np.mean(np.array(rew_arr))


if __name__ == "__main__":
    fight = Fight()
    fight.train_model(print=True)
