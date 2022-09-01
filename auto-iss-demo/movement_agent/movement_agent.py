import os
from collections import deque, namedtuple

from movement_enviroment import movement_enviroment
import csv

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])
AuxMemory = namedtuple('Memory', ['state', 'target_value', 'old_values'])

class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

# helpers

def exists(val):
    return val is not None

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

def init_(m):
    if isinstance(m, nn.Linear):
        gain = torch.nn.init.calculate_gain('tanh')
        torch.nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# networks

class Actor(nn.Module):
    def __init__(self, state_dim,num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.action_head = nn.Sequential(
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(64, 1)
        self.apply(init_)

    def forward(self, x):
        hidden = self.net(x)
        return self.action_head(hidden), self.value_head(hidden)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        self.apply(init_)

    def forward(self, x):
        return self.net(x)

# agent

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

class PPG:
    def __init__(
        self,
        save_path = 'PPG_net',
        state_dim = 3,
        num_actions = 3,
        epochs = 1,
        epochs_aux = 9,
        minibatch_size = 64, 
        lr = 0.0005,
        lam = 0.95,
        gamma = 0.99,
        beta_s = .01,
        eps_clip = 0.2,
        value_clip = 0.4,
    ):
        self.actor = Actor(state_dim, num_actions).to(device)
        self.critic = Critic(state_dim).to(device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr)

        self.minibatch_size = minibatch_size

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.save_path = save_path

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, f'./'+ self.save_path + '.pt')

    def load(self,path):
        if not os.path.exists(path):
            print('El archivo seleccionado: '+path+' no existe')
            return
        
        print('Agente cardado correctamente')
        data = torch.load(f'./'+ path)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        
        
    def display_info(self,eps,steps,max_steps,last_reward,state,first_state):
        
        # Distancia euclidea      
        
        x_si = str(first_state[0])[:-2]
        y_si = str(first_state[1])[:-2]
        z_si = str(first_state[2])[:-2]

        x_sf = str(state[0])[:-2]
        y_sf = str(state[1])[:-2]
        z_sf = str(state[2])[:-2]

            
        print('Episode '+str(eps)+' |Steps = ' + str(steps+1) +'/'+ str(max_steps) + ' | Rewards = [' + str(last_reward)[:4] + ']' + ' | Initial State: ' + '['+x_si+','+y_si+','+z_si+']' + ' | Final State: ' + '['+x_sf+','+y_sf+','+z_sf+']') 
            
        return 
        

    def learn(self, memories, aux_memories, next_state):
        # retrieve and prepare data from memory for training
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in memories:
            states.append(torch.tensor(mem.state))
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        # calculate generalized advantage estimate
        next_state = torch.from_numpy(np.asarray(next_state)).float()
        next_value = self.critic(next_state).detach()
        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # convert values to torch tensors
        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values[:-1])
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(returns).to(device)

        # store state and target values to auxiliary memory buffer for later training
        aux_memory = AuxMemory(states, rewards, old_values)
        aux_memories.append(aux_memory)

        # prepare dataloader for policy phase training
        dl = create_shuffled_dataloader([states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        # policy phase training, similar to original PPO
        for _ in range(self.epochs):
            for states, actions, old_log_probs, rewards, old_values in dl:
                action_probs, _ = self.actor(states)
                values = self.critic(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # calculate clipped surrogate objective, classic PPO loss
                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                update_network_(policy_loss, self.opt_actor)

                # calculate value loss and update value network separate from policy network
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)

    def learn_aux(self, aux_memories):
        # gather states and target values into one tensor
        states = []
        rewards = []
        old_values = []
        for state, reward, old_value in aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)

        states = torch.cat(states)
        rewards = torch.cat(rewards)
        old_values = torch.cat(old_values)

        # get old action predictions for minimizing kl divergence and clipping respectively
        old_action_probs, _ = self.actor(states)
        old_action_probs.detach_()

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader([states, old_action_probs, rewards, old_values], self.minibatch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)
        for epoch in range(self.epochs_aux):
            for states, old_action_probs, rewards, old_values in dl:
                action_probs, policy_values = self.actor(states)
                action_logprobs = action_probs.log()

                # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                policy_loss = aux_loss + loss_kl

                update_network_(policy_loss, self.opt_actor)

                # paper says it is important to train the value network extra during the auxiliary phase
                values = self.critic(states)
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)
                           

    def train(self,num_episodes = 50000,max_steps = 500,update_steps = 5000,
              num_policy_updates_per_aux = 32,seed = None, save_every = 1000):

        # Load enviroment and buffers
        
        env = movement_enviroment()

        memories = deque([])
        aux_memories = deque([])

        if exists(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)

        num_policy_updates = 0
        total_steps = 0

        for eps in range(1,num_episodes+1):

            env.reset()
            state = env.state
            first_state = [state[0],state[1],state[2]]
            
            for steps in range(max_steps):

                # Perform rollouts
                total_steps += 1

                state_tensor = torch.from_numpy(np.asarray(state)).float()
                action_probs, _ = self.actor(state_tensor)
                value = self.critic(state_tensor)
                
                dist = Categorical(action_probs)
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                action = action.item()

                next_state, reward, done = env.train_step(action)                
                state = next_state
                
                memory = Memory(state, action, action_log_prob, reward, done, value)
                memories.append(memory)

                # Update Policy and Value network

                if total_steps % update_steps == 0:   
                    self.display_info(eps,steps,max_steps,reward,state,first_state)
                    self.learn(memories, aux_memories,next_state)
                    num_policy_updates += 1
                    memories.clear()

                    # Update Auxiliar network

                    if num_policy_updates % num_policy_updates_per_aux == 0:
                        self.learn_aux(aux_memories)
                        aux_memories.clear()

                # End of epochs via failure/success or max_steps 

                if done:
                    break
                if total_steps % max_steps == 0:
                    break        
                

            if eps % save_every == 0:
                self.save()

        self.save()    
        
    def test(self,max_episodes,max_steps,save_csv):

        # Load enviroment and buffers
        
        env = movement_enviroment()
        success = 0
        
        with open(str(save_csv) + '.csv','a+',newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(['Episodde','Step','X','Y','Z'])            
            file.close()
        
        for episode in range(max_episodes):
            print(str(episode),end = '\r')
            env.test_state()
            state = env.state
            with open(str(save_csv) + '.csv','a+',newline = '') as file:
                writer = csv.writer(file)
                writer.writerow([episode,0,state[0],state[1],state[2]])          
                file.close()                 
                             
            status = []

            for test_steps in range(max_steps):

                state_tensor = torch.from_numpy(np.asarray(state)).float()
                action_probs, _ = self.actor(state_tensor)

                dist = Categorical(action_probs)
                action = dist.sample()
                action = action.item()

                next_state, done = env.test_step(action)    
                state = next_state
                
                status.append([episode,test_steps+1,state[0],state[1],state[2]])

                if state[0] == 6 and state[1] == 0 and state[2] == 0:
                        success += 1 

                if done == True:
                    with open(str(save_csv) + '.csv','a+',newline = '') as file:
                        writer = csv.writer(file)
                        for x in range(len(status)):
                            writer.writerow([status[x][0],status[x][1],status[x][2],status[x][3],status[x][4]])
                        file.close()
                    break 

                if test_steps+1 == max_steps:
                    with open(str(save_csv) + '.csv','a+',newline = '') as file:
                        writer = csv.writer(file)
                        for x in range(len(status)):
                            writer.writerow([status[x][0],status[x][1],status[x][2],status[x][3],status[x][4]])
                        file.close()
                    break
        
        print('Resultados del test guardados en: '+ str(save_csv) + '.csv')
        print('Realizando ' + str(max_episodes) + ' episodios de '+ str(max_steps) + ' steps el agente tiene un porcentaje de éxito del ' + str((success/max_episodes)*100) + '%')
       