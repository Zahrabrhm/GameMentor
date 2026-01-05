#!/usr/bin/env python

import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
import h5py
import warnings
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):
    def __init__(self, layers=[8, 64, 32, 4], dropout=False, p_dropout=0.5):
        super(NeuralNetwork, self).__init__()
        self.network_layers = []
        n_layers = len(layers)
        
        for i, neurons_in_current_layer in enumerate(layers[:-1]):
            self.network_layers.append(nn.Linear(neurons_in_current_layer, layers[i+1]))
            if dropout:
                self.network_layers.append(nn.Dropout(p=p_dropout))
            if i < n_layers - 2:
                self.network_layers.append(nn.ReLU())
        
        self.network_layers = nn.Sequential(*self.network_layers)

    def forward(self, x):
        for layer in self.network_layers:
            x = layer(x)
        return x


class AgentBase:
    def __init__(self, parameters):
        parameters = self._make_dict_keys_lowercase(parameters)
        self._set_initialization_parameters(parameters)
        default_parameters = self._get_default_parameters()
        parameters = self._merge_dictionaries(parameters, default_parameters)
        self._set_parameters(parameters)
        self.parameters = copy.deepcopy(parameters)
        self._initialize_neural_networks(parameters['neural_networks'])
        self._initialize_optimizers(parameters['optimizers'])
        self._initialize_losses(parameters['losses'])
        self.in_training = False

    def save_weights(self, filename):
        policy_net_state_dict = self.neural_networks['policy_net'].state_dict()
        target_net_state_dict = self.neural_networks['target_net'].state_dict()
        torch.save({
            'policy_net_state_dict': policy_net_state_dict,
            'target_net_state_dict': target_net_state_dict
        }, filename)

    def load_weights(self, filename):
        checkpoint = torch.load(filename, map_location=device)
        self.neural_networks['policy_net'].load_state_dict(checkpoint['policy_net_state_dict'])
        self.neural_networks['target_net'].load_state_dict(checkpoint['target_net_state_dict'])

    def _make_dict_keys_lowercase(self, dictionary):
        return {key.lower(): value for key, value in dictionary.items()}

    def _merge_dictionaries(self, dict1, dict2):
        return_dict = copy.deepcopy(dict1)
        for key, value in dict2.items():
            if key not in return_dict:
                return_dict[key] = value
        return return_dict

    def _get_default_parameters(self):
        return {
            'neural_networks': {
                'policy_net': {'layers': [self.n_state, 128, 32, self.n_actions]}
            },
            'optimizers': {
                'policy_net': {'optimizer': 'RMSprop', 'optimizer_args': {'lr': 1e-3}}
            },
            'losses': {
                'policy_net': {'loss': 'MSELoss'}
            },
            'n_memory': 20000,
            'training_stride': 5,
            'batch_size': 32,
            'saving_stride': 100,
            'n_episodes_max': 10000,
            'n_solving_episodes': 20,
            'solving_threshold_min': 200,
            'solving_threshold_mean': 230,
            'discount_factor': 0.99,
        }

    def _set_initialization_parameters(self, parameters):
        try:
            self.n_state = parameters['n_state']
        except KeyError:
            raise RuntimeError("Parameter N_state is required.")
        try:
            self.n_actions = parameters['n_actions']
        except KeyError:
            raise RuntimeError("Parameter N_actions is required.")

    def _set_parameters(self, parameters):
        parameters = self._make_dict_keys_lowercase(parameters)
        
        if 'discount_factor' in parameters:
            self.discount_factor = parameters['discount_factor']
        if 'n_memory' in parameters:
            self.n_memory = int(parameters['n_memory'])
            self.memory = ReplayMemory(self.n_memory)
        if 'training_stride' in parameters:
            self.training_stride = parameters['training_stride']
        if 'batch_size' in parameters:
            self.batch_size = int(parameters['batch_size'])
        if 'saving_stride' in parameters:
            self.saving_stride = parameters['saving_stride']
        if 'n_episodes_max' in parameters:
            self.n_episodes_max = parameters['n_episodes_max']
        if 'n_solving_episodes' in parameters:
            self.n_solving_episodes = parameters['n_solving_episodes']
        if 'solving_threshold_min' in parameters:
            self.solving_threshold_min = parameters['solving_threshold_min']
        if 'solving_threshold_mean' in parameters:
            self.solving_threshold_mean = parameters['solving_threshold_mean']

    def get_parameters(self):
        return self.parameters

    def _initialize_neural_networks(self, neural_networks):
        self.neural_networks = {}
        for key, value in neural_networks.items():
            self.neural_networks[key] = NeuralNetwork(value['layers']).to(device)

    def _initialize_optimizers(self, optimizers):
        self.optimizers = {}
        for key, value in optimizers.items():
            self.optimizers[key] = torch.optim.RMSprop(
                self.neural_networks[key].parameters(),
                **value['optimizer_args']
            )

    def _initialize_losses(self, losses):
        self.losses = {}
        for key, value in losses.items():
            self.losses[key] = nn.MSELoss()

    def get_number_of_model_parameters(self, name='policy_net'):
        return sum(p.numel() for p in self.neural_networks[name].parameters() if p.requires_grad)

    def get_state(self):
        state = {'parameters': self.get_parameters()}
        for name, neural_network in self.neural_networks.items():
            state[name] = copy.deepcopy(neural_network.state_dict())
        for name, optimizer in self.optimizers.items():
            state[name + '_optimizer'] = copy.deepcopy(optimizer.state_dict())
        return state

    def load_state(self, state):
        parameters = state['parameters']
        self._check_parameter_compatibility(parameters)
        self.__init__(parameters=parameters)
        
        for name, state_dict in state.items():
            if name == 'parameters':
                continue
            elif 'optimizer' in name:
                name = name.replace('_optimizer', '')
                self.optimizers[name].load_state_dict(state_dict)
            else:
                self.neural_networks[name].load_state_dict(state_dict)

    def _check_parameter_compatibility(self, parameters):
        if 'n_state' in parameters and parameters['n_state'] != self.n_state:
            raise RuntimeError("Incompatible n_state parameter.")
        if 'n_actions' in parameters and parameters['n_actions'] != self.n_actions:
            raise RuntimeError("Incompatible n_actions parameter.")

    def evaluate_stopping_criterion(self, list_of_returns):
        if len(list_of_returns) < self.n_solving_episodes:
            return False, 0., 0.
        
        recent_returns = np.array(list_of_returns)[-self.n_solving_episodes:]
        minimal_return = np.min(recent_returns)
        mean_return = np.mean(recent_returns)
        
        if minimal_return > self.solving_threshold_min and mean_return > self.solving_threshold_mean:
            return True, minimal_return, mean_return
        return False, minimal_return, mean_return

    def act(self, state):
        return np.random.randint(self.n_actions)

    def add_memory(self, memory):
        self.memory.push(*memory)

    def get_samples_from_memory(self):
        current_transitions = self.memory.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*current_transitions))
        
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0)
        next_state_batch = torch.cat([s.unsqueeze(0) for s in batch.next_state], dim=0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.tensor(batch.done).float()
        
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

    def run_optimization_step(self, epoch):
        pass

    def train(self, environment, verbose=True, model_filename=None, training_filename=None):
        self.in_training = True
        training_complete = False
        step_counter = 0
        epoch_counter = 0
        
        episode_durations = []
        episode_returns = []
        steps_simulated = []
        training_epochs = []
        output_state_dicts = {}

        if verbose:
            print("Training started...")

        for n_episode in range(self.n_episodes_max):
            state, info = environment.reset()
            current_total_reward = 0.

            for i in itertools.count():
                action = self.act(state=state)
                next_state, reward, terminated, truncated, info = environment.step(action)
                step_counter += 1
                done = terminated or truncated
                current_total_reward += reward

                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)
                self.add_memory([
                    torch.tensor(state),
                    action,
                    torch.tensor(next_state),
                    reward,
                    done
                ])

                state = next_state

                if step_counter % self.training_stride == 0:
                    self.run_optimization_step(epoch=epoch_counter)
                    epoch_counter += 1

                if done:
                    episode_durations.append(i + 1)
                    episode_returns.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_epochs.append(epoch_counter)

                    training_complete, min_ret, mean_ret = self.evaluate_stopping_criterion(episode_returns)
                    
                    if verbose and n_episode % 100 == 0:
                        print(f"Episode {n_episode}: Return={current_total_reward:.2f}, Mean={mean_ret:.2f}")
                    break

            if (n_episode % self.saving_stride == 0) or training_complete or n_episode == self.n_episodes_max - 1:
                if model_filename:
                    output_state_dicts[n_episode] = self.get_state()
                    torch.save(output_state_dicts, model_filename)

            if training_complete:
                break

        if not training_complete:
            warnings.warn("Training stopped at max episodes without meeting stopping criterion.")

        self.in_training = False
        return {
            'episode_durations': episode_durations,
            'episode_returns': episode_returns,
            'n_training_epochs': training_epochs,
            'n_steps_simulated': steps_simulated,
            'training_completed': training_complete
        }


class DQN(AgentBase):
    def __init__(self, parameters):
        super().__init__(parameters=parameters)
        self.in_training = False

    def _get_default_parameters(self):
        default_parameters = super()._get_default_parameters()
        default_parameters['neural_networks']['target_net'] = {
            'layers': copy.deepcopy(default_parameters['neural_networks']['policy_net']['layers'])
        }
        default_parameters['target_net_update_stride'] = 1
        default_parameters['target_net_update_tau'] = 1e-2
        default_parameters['epsilon'] = 1.0
        default_parameters['epsilon_1'] = 0.1
        default_parameters['d_epsilon'] = 0.00005
        default_parameters['doubledqn'] = False
        return default_parameters

    def _set_parameters(self, parameters):
        super()._set_parameters(parameters)
        
        if 'doubledqn' in parameters:
            self.doubleDQN = parameters['doubledqn']
        if 'target_net_update_stride' in parameters:
            self.target_net_update_stride = parameters['target_net_update_stride']
        if 'target_net_update_tau' in parameters:
            self.target_net_update_tau = parameters['target_net_update_tau']
        if 'epsilon' in parameters:
            self.epsilon = parameters['epsilon']
        if 'epsilon_1' in parameters:
            self.epsilon_1 = parameters['epsilon_1']
        if 'd_epsilon' in parameters:
            self.d_epsilon = parameters['d_epsilon']

    def act(self, state, epsilon=0.):
        if self.in_training:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            policy_net = self.neural_networks['policy_net']
            with torch.no_grad():
                policy_net.eval()
                action = policy_net(torch.tensor(state)).argmax(0).item()
                policy_net.train()
                return action
        else:
            return torch.randint(low=0, high=self.n_actions, size=(1,)).item()

    def act_values(self, state):
        policy_net = self.neural_networks['policy_net']
        with torch.no_grad():
            policy_net.eval()
            return policy_net(torch.tensor(state))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

    def run_optimization_step(self, epoch):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.get_samples_from_memory()

        policy_net = self.neural_networks['policy_net']
        target_net = self.neural_networks['target_net']
        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net']

        policy_net.train()

        LHS = policy_net(state_batch.to(device)).gather(dim=1, index=action_batch.unsqueeze(1))

        if self.doubleDQN:
            argmax_next_state = policy_net(next_state_batch).argmax(dim=1)
            Q_next_state = target_net(next_state_batch).gather(
                dim=1, index=argmax_next_state.unsqueeze(1)
            ).squeeze(1)
        else:
            Q_next_state = target_net(next_state_batch).max(1)[0].detach()

        RHS = Q_next_state * self.discount_factor * (1. - done_batch) + reward_batch
        RHS = RHS.unsqueeze(1)

        loss_ = loss(LHS, RHS)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        policy_net.eval()
        self.update_epsilon()

        if epoch % self.target_net_update_stride == 0:
            self._soft_update_target_net()

    def _soft_update_target_net(self):
        params1 = self.neural_networks['policy_net'].named_parameters()
        params2 = self.neural_networks['target_net'].named_parameters()
        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(
                    self.target_net_update_tau * param1.data +
                    (1 - self.target_net_update_tau) * dict_params2[name1].data
                )
        self.neural_networks['target_net'].load_state_dict(dict_params2)


class ActorCritic(AgentBase):
    def __init__(self, parameters):
        super().__init__(parameters=parameters)
        self.Softmax = nn.Softmax(dim=0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def _get_default_parameters(self):
        default_parameters = super()._get_default_parameters()
        default_parameters['neural_networks']['critic_net'] = {
            'layers': [self.n_state, 64, 32, 1]
        }
        default_parameters['optimizers']['critic_net'] = {
            'optimizer': 'RMSprop',
            'optimizer_args': {'lr': 1e-3}
        }
        default_parameters['affinities_regularization'] = 0.01
        return default_parameters

    def _set_parameters(self, parameters):
        super()._set_parameters(parameters)
        if 'affinities_regularization' in parameters:
            self.affinities_regularization = parameters['affinities_regularization']

    def _initialize_losses(self, losses):
        def loss_actor(state_batch, action_batch, advantage_batch):
            affinities = self.neural_networks['policy_net'](state_batch)
            log_pi_a = self.LogSoftmax(affinities).gather(dim=1, index=action_batch.unsqueeze(1))
            loss_actor = -log_pi_a * advantage_batch + \
                         self.affinities_regularization * torch.sum(affinities**2) / self.batch_size
            return loss_actor.sum()

        self.losses = {
            'policy_net': loss_actor,
            'critic_net': nn.MSELoss()
        }

    def act(self, state):
        actor_net = self.neural_networks['policy_net']
        with torch.no_grad():
            actor_net.eval()
            probs = self.Softmax(actor_net(torch.tensor(state)))
            m = Categorical(probs)
            action = m.sample()
            actor_net.train()
            return action.item()

    def run_optimization_step(self, epoch):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.get_samples_from_memory()

        actor_net = self.neural_networks['policy_net']
        critic_net = self.neural_networks['critic_net']
        optimizer_actor = self.optimizers['policy_net']
        optimizer_critic = self.optimizers['critic_net']
        loss_actor = self.losses['policy_net']
        loss_critic = self.losses['critic_net']

        critic_net.train()
        LHS = critic_net(state_batch.to(device))
        Q_next_state = critic_net(next_state_batch).detach().squeeze(1)
        RHS = Q_next_state * self.discount_factor * (1. - done_batch) + reward_batch
        RHS = RHS.unsqueeze(1)

        loss = loss_critic(LHS, RHS)
        optimizer_critic.zero_grad()
        loss.backward()
        optimizer_critic.step()
        critic_net.eval()

        actor_net.train()
        advantage_batch = (RHS - LHS).detach()
        loss = loss_actor(state_batch, action_batch, advantage_batch)
        optimizer_actor.zero_grad()
        loss.backward()
        optimizer_actor.step()
        actor_net.eval()
