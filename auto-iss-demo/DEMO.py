import os
from collections import deque, namedtuple
import time
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

from matplotlib.style import available
import selenium
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from torch import double
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import numpy as np


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
                           


class enviroment():

    def __init__(self,localhost = 5555):

        # Create the driver with selenium

        print('[INFO] Conectando con chrome...')
        chrome_options = Options()
        chrome_options.add_argument('log-level=0')
        chrome_options.add_experimental_option("detach",True)
        driver = webdriver.Chrome(options=chrome_options,service=Service(ChromeDriverManager().install())) 

        driver.get("http://localhost:" + str(localhost) + "/iss-sim.spacex.com")
        self.driver = driver

        # Find login button
        login_button = driver.find_element(by=By.ID,value = 'begin-button')
        
        print('[INFO] Conexión establecida')

        # Click login
        while True:
            if login_button.is_displayed():
                login_button.click()
                time.sleep(10)
                break
        

    def rotational_state(self):
 
        roll = float(self.driver.find_element(by=By.ID,value = 'roll').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])
        pitch = float(self.driver.find_element(by=By.ID,value = 'pitch').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])
        yaw = float(self.driver.find_element(by=By.ID,value = 'yaw').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])

        return [roll,pitch,yaw]


    def movement_state(self):

        x = float(self.driver.find_element(by=By.ID,value = 'x-range').get_attribute('innerText')[:-1:])
        y = float(self.driver.find_element(by=By.ID,value = 'y-range').get_attribute('innerText')[:-1:])
        z = float(self.driver.find_element(by=By.ID,value = 'z-range').get_attribute('innerText')[:-1:])

        return [x,y,z]

    def reset(self):
        restart_button = self.driver.find_element(by=By.ID,value = 'option-restart')
        restart_button.click()
        time.sleep(7)
        return 

    def restart(self):
        if self.success():
            restart_button = self.driver.find_element(by=By.ID,value = 'success-button')
        else:
            restart_button = self.driver.find_element(by=By.ID,value = 'fail-button')
        time.sleep(2)
        restart_button.click()
        time.sleep(8)
        return 

    def fail(self):
        fail_button = self.driver.find_element(by=By.ID,value = 'fail-button')
        if self.HUD_available() == False:
            if fail_button.is_displayed() == True:
                return True
        else:
            return False

    def success(self):
        success_button = self.driver.find_element(by=By.ID,value = 'success-button')
        if self.HUD_available() == False:
            if success_button.is_displayed() == True:
                return True
        else:
            return False
        
    def action(self,action_number):
        if action_number == 0:
            action_button = self.driver.find_element(by=By.ID,value = 'translate-left-button')
        if action_number == 1:
            action_button = self.driver.find_element(by=By.ID,value = 'translate-right-button')
        if action_number == 2:
            action_button = self.driver.find_element(by=By.ID,value = 'translate-up-button')
        if  action_number == 3:
            action_button = self.driver.find_element(by=By.ID,value = 'translate-down-button')
        if action_number == 4:
            action_button = self.driver.find_element(by=By.ID,value = 'translate-forward-button')
        if action_number == 5:
            action_button = self.driver.find_element(by=By.ID,value = 'translate-backward-button')
        if action_number == 6:
            action_button = self.driver.find_element(by=By.ID,value = 'yaw-left-button')
        if action_number == 7:
            action_button = self.driver.find_element(by=By.ID,value = 'yaw-right-button')
        if action_number == 8:
            action_button = self.driver.find_element(by=By.ID,value = 'pitch-up-button')
        if action_number == 9:
            action_button = self.driver.find_element(by=By.ID,value = 'pitch-down-button')
        if action_number == 10:
            action_button = self.driver.find_element(by=By.ID,value = 'roll-left-button')
        if action_number == 11:
            action_button = self.driver.find_element(by=By.ID,value = 'roll-right-button')
        if action_number == 12:
            return       

        action_button.click()

        return

    def close(self):
        self.driver.close()

    def HUD_available(self):
        
        x = self.driver.find_element(by=By.ID,value = 'x-range').get_attribute('innerText')[:-1:]
        y = self.driver.find_element(by=By.ID,value = 'y-range').get_attribute('innerText')[:-1:]
        z = self.driver.find_element(by=By.ID,value = 'z-range').get_attribute('innerText')[:-1:]
        yaw = self.driver.find_element(by=By.ID,value = 'yaw').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:]
        roll = self.driver.find_element(by=By.ID,value = 'roll').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:]
        pitch = self.driver.find_element(by=By.ID,value = 'pitch').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:]
        yaw_v = self.driver.find_element(by=By.ID,value = 'yaw').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:]
        roll_v = self.driver.find_element(by=By.ID,value = 'roll').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:]
        pitch_v = self.driver.find_element(by=By.ID,value = 'pitch').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:]
        xyz_range = self.driver.find_element(by=By.ID,value = 'range').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-1:] 
        xyz_rate = self.driver.find_element(by=By.ID,value = 'rate').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:] 
        
        if x == '' or y == '' or z == '' or yaw == '' or roll == '' or pitch == '' or yaw_v == '' or roll_v == '' or pitch_v == '' or xyz_range == '' or xyz_rate == '':
            return False
        else:
            return True

    def generate_rotations(self,action_number,state):

        if state[action_number] > 0:
            state[action_number] -= 1
        else:
            state[action_number] += 1

        if state[0] == 0 and state[1] == 0 and state[2] == 0:
            return state,True
        else:
            return state,False

    def generate_traslations(self,action_number,state):

        if state[action_number] > 0:
            state[action_number] -= 1
        else:
            state[action_number] += 1
            
        if  state[0] < 6:
            return state,True        
            
        if state[0] == 6 and state[1] == 0 and state[2] == 0:
            return state,True

        return  state,False
            
    def perform_rotation(self,rotation_list):

        roll_actions = rotation_list.count(0)
        pitch_actions = rotation_list.count(1)
        yaw_actions = rotation_list.count(2)

        state = preprocess_action(self.rotational_state())
        roll = state[0]
        pitch = state[1]
        yaw = state[2]

        new_state = preprocess_action(self.rotational_state())
        new_roll = new_state[0]
        new_pitch = new_state[1]
        new_yaw = new_state[2]

        if roll_actions != 0 :
            for _ in range(roll_actions):          
                if roll < 0: 
                    R = new_roll+1
                    action_button = self.driver.find_element(by=By.ID,value = 'roll-left-button')
                    action_button.click()
                    while R > new_roll:
                        new_state = preprocess_action(self.rotational_state())
                        new_roll = new_state[0]
                    action_button = self.driver.find_element(by=By.ID,value = 'roll-right-button')
                    action_button.click()
                else:
                    R = new_roll-1
                    action_button = self.driver.find_element(by=By.ID,value = 'roll-right-button')
                    action_button.click()
                    while R < new_roll:
                        new_state = preprocess_action(self.rotational_state())
                        new_roll = new_state[0]                    
                    action_button = self.driver.find_element(by=By.ID,value = 'roll-left-button')
                    action_button.click()
                state = preprocess_action(self.rotational_state())
                roll = state[0]

        if pitch_actions != 0 :
            for _ in range(pitch_actions):          
                if pitch < 0: 
                    P = new_pitch+1
                    action_button = self.driver.find_element(by=By.ID,value = 'pitch-up-button')
                    action_button.click()
                    while P > new_pitch:
                        new_state = preprocess_action(self.rotational_state())
                        new_pitch = new_state[1]
                    action_button = self.driver.find_element(by=By.ID,value = 'pitch-down-button')
                    action_button.click()
                else:
                    P = new_pitch-1
                    action_button = self.driver.find_element(by=By.ID,value = 'pitch-down-button')
                    action_button.click()
                    while P < new_pitch:
                        new_state = preprocess_action(self.rotational_state())
                        new_pitch = new_state[1]                    
                    action_button = self.driver.find_element(by=By.ID,value = 'pitch-up-button')
                    action_button.click()
                state = preprocess_action(self.rotational_state())
                pitch = state[1]

        if yaw_actions != 0 :
            for _ in range(yaw_actions):          
                if yaw < 0: 
                    Y = new_yaw+1
                    action_button = self.driver.find_element(by=By.ID,value = 'yaw-left-button')
                    action_button.click()
                    while Y > new_yaw:
                        new_state = preprocess_action(self.rotational_state())
                        new_yaw = new_state[2]
                    action_button = self.driver.find_element(by=By.ID,value = 'yaw-right-button')
                    action_button.click()
                else:
                    Y = new_yaw-1
                    action_button = self.driver.find_element(by=By.ID,value = 'yaw-right-button')
                    action_button.click()
                    while Y < new_yaw:
                        new_state = preprocess_action(self.rotational_state())
                        new_yaw = new_state[2]                    
                    action_button = self.driver.find_element(by=By.ID,value = 'yaw-left-button')
                    action_button.click()
                state = preprocess_action(self.rotational_state())
                yaw = state[2]

    def perform_movement(self,movement_list):

        state = preprocess_action(self.movement_state())
        y = state[1]
        z = state[2]

        new_state = preprocess_action(self.movement_state())
        new_x = new_state[0]
        new_y = new_state[1]
        new_z = new_state[2]

        for action in movement_list:
            print('Action = ' + str(action))
            if action == 0:
                    X = new_x-1
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-forward-button')
                    action_button.click()
                    print('X = ' + str(X+1))
                    while X < new_x:
                        new_state = preprocess_action(self.movement_state())
                        new_x = new_state[0]
                    print('new_X = ' + str(new_x))
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-backward-button')
                    action_button.click()
                    state = preprocess_action(self.movement_state())
                    self.calibrate_Z()

            if action == 1:         
                if y < 0: 
                    Y = new_y+1
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-right-button')
                    action_button.click()
                    print('Y = ' + str(Y-1))
                    while Y > new_y:
                        new_state = preprocess_action(self.movement_state())
                        new_y = new_state[1]
                    print('new_y = ' + str(new_y))
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-left-button')
                    action_button.click()
                    
                else:
                    Y = new_y-1
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-left-button')
                    action_button.click()
                    print('Y = ' + str(Y+1))
                    while Y < new_y:
                        new_state = preprocess_action(self.movement_state())
                        new_y = new_state[1] 
                    print('new_y = ' + str(new_y))                 
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-right-button')
                    action_button.click()
                state = preprocess_action(self.movement_state())
                y = state[1]

            if action == 2:         
                if z < 0: 
                    Z = new_z+1
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-up-button')
                    action_button.click()
                    print('Z = ' + str(Z-1))
                    while Z > new_z:
                        new_state = preprocess_action(self.movement_state())
                        new_z = new_state[2]
                    print('new_z = ' + str(new_z))    
                   
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-down-button')
                    action_button.click()
                else:
                    Z = new_z-1
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-down-button')
                    action_button.click()
                    print('Z = ' + str(Z+1))
                    while Z < new_z:
                        new_state = preprocess_action(self.movement_state())
                        new_z = new_state[2]  
                    print('new_z = ' + str(new_z))                  
                    action_button = self.driver.find_element(by=By.ID,value = 'translate-up-button')
                    action_button.click()
                state = preprocess_action(self.movement_state())
                z = state[2]

    def calibrate_Z(self):
        print('Calibrate')
        state = preprocess_action(self.movement_state())
        z = state[2]
        new_state = preprocess_action(self.movement_state())
        new_z = new_state[2]  

        if z < 0:
            if np.abs(z) - np.abs(int(z)) > 0.5:
                Z = int(new_z - 1)
                action_button = self.driver.find_element(by=By.ID,value = 'translate-down-button')
                action_button.click()
                print('Z_R = ' + str(Z-1))
                while Z < new_z:
                    new_state = preprocess_action(self.movement_state())
                    new_z = new_state[2]  
                print('new_z_R = ' + str(new_z))                  
                action_button = self.driver.find_element(by=By.ID,value = 'translate-up-button')
                action_button.click()
            else:
                Z = int(new_z)
                action_button = self.driver.find_element(by=By.ID,value = 'translate-up-button')
                action_button.click()
                print('Z_R = ' + str(Z))
                while Z > new_z:
                    new_state = preprocess_action(self.movement_state())
                    new_z = new_state[2]  
                print('new_z_R = ' + str(new_z))                  
                action_button = self.driver.find_element(by=By.ID,value = 'translate-down-button')
                action_button.click()
        else:
            if np.abs(z) - np.abs(int(z)) > 0.5:
                Z = int(new_z + 1)
                action_button = self.driver.find_element(by=By.ID,value = 'translate-up-button')
                action_button.click()
                print('Z_R = ' + str(Z+1))
                while Z > new_z:
                    new_state = preprocess_action(self.movement_state())
                    new_z = new_state[2]  
                print('new_z_R = ' + str(new_z))                  
                action_button = self.driver.find_element(by=By.ID,value = 'translate-down-button')
                action_button.click()
            else:
                Z = int(new_z)
                action_button = self.driver.find_element(by=By.ID,value = 'translate-down-button')
                action_button.click()
                print('Z_R = ' + str(Z))
                while Z < new_z:
                    new_state = preprocess_action(self.movement_state())
                    new_z = new_state[2]  
                print('new_z_R = ' + str(new_z))                  
                action_button = self.driver.find_element(by=By.ID,value = 'translate-up-button')
                action_button.click()

        return  
                
    def perform_docking(self):

        new_state = preprocess_action(self.movement_state())
        new_x = new_state[0]

        X = new_x-0.6
        action_button = self.driver.find_element(by=By.ID,value = 'translate-forward-button')
        action_button.click()
        while X < new_x:
            new_state = preprocess_action(self.movement_state())
            new_x = new_state[0]
        action_button = self.driver.find_element(by=By.ID,value = 'translate-backward-button')
        action_button.click()
        state = preprocess_action(self.movement_state())
        self.calibrate_Z()

        X = new_x-0.2
        action_button = self.driver.find_element(by=By.ID,value = 'translate-forward-button')
        action_button.click()
        while X < new_x:
            new_state = preprocess_action(self.movement_state())
            new_x = new_state[0]
        action_button = self.driver.find_element(by=By.ID,value = 'translate-backward-button')
        action_button.click()
        state = preprocess_action(self.movement_state())
        self.calibrate_Z()

        X = new_x-0.1
        action_button = self.driver.find_element(by=By.ID,value = 'translate-forward-button')
        action_button.click()
        while X < new_x:
            new_state = preprocess_action(self.movement_state())
            new_x = new_state[0]
        action_button = self.driver.find_element(by=By.ID,value = 'translate-backward-button')
        action_button.click()
        state = preprocess_action(self.movement_state())
        self.calibrate_Z()

        action_button = self.driver.find_element(by=By.ID,value = 'translate-forward-button')
        action_button.click()



def preprocess_state(state):
    return [int(np.abs(state[0]/10)),int(np.abs(state[1]/10)),int(np.abs(state[2]/10))]

def preprocess_action(state):
    return [np.round(state[0]/10,2),np.round(state[1]/10,2),np.round(state[2]/10,2)]


def play_demo(rotational_agent,movement_agent,env_name = 5555):

    # Conect with the enviroment

    env = enviroment(env_name)
    done = False

    # Make a rotation trayectory that succeds

    state_r = preprocess_state(env.rotational_state())

    if state_r[0] == 0 and state_r[1] == 0 and state_r[2] == 0:
        done == True
    else:
        while done == False:

            rotations = []
            next_state = state_r.copy()

            for _ in range(15):
            # Perform rollouts
                state_tensor = torch.from_numpy(np.asarray(next_state)).float()
                action_probs, _ = rotational_agent.actor(state_tensor)

                dist = Categorical(action_probs)
                action = dist.sample()
                action = action.item()
                rotations.append(action)

                next_state,done = env.generate_rotations(action,next_state)
                if done:
                    print('Rotations')
                    print(rotations)
                    break

        env.perform_rotation(rotations)

    # Make a movement trayectory that succeds
    state_m = preprocess_state(env.movement_state())

    env.calibrate_Z()

    if state_m[0] == 6 and state_m[1] == 0 and state_m[2] == 0:
        done == True
    else:
        done = False
        while done == False:

            movement = []
            next_state = state_m.copy()

            for _ in range(50):
            # Perform rollouts
                state_tensor = torch.from_numpy(np.asarray(next_state)).float()
                action_probs, _ = movement_agent.actor(state_tensor)

                dist = Categorical(action_probs)
                action = dist.sample()
                action = action.item()
                movement.append(action)

                next_state,done = env.generate_traslations(action,next_state)

                if done:
                    if next_state[0] == 6 and next_state[1] == 0 and next_state[2] == 0:
                        print('Movement: ')
                        print(movement)
                        break
                    else:
                        done = False
                        break

        # Perform rotation strategy

        env.perform_movement(movement)

    # Perform docking manouver
    env.perform_movement([0,0,0,0,0])
    env.calibrate_Z()
    env.perform_docking()
        
    while True:
        print(1,end = '\r')