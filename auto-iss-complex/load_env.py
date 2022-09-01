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
        

    def state(self):

        x = float(self.driver.find_element(by=By.ID,value = 'x-range').get_attribute('innerText')[:-1:])
        y = float(self.driver.find_element(by=By.ID,value = 'y-range').get_attribute('innerText')[:-1:])
        z = float(self.driver.find_element(by=By.ID,value = 'z-range').get_attribute('innerText')[:-1:])
        yaw = float(self.driver.find_element(by=By.ID,value = 'yaw').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:]) 
        roll = float(self.driver.find_element(by=By.ID,value = 'roll').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])
        pitch = float(self.driver.find_element(by=By.ID,value = 'pitch').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])
        yaw_v = float(self.driver.find_element(by=By.ID,value = 'yaw').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:])
        roll_v = float(self.driver.find_element(by=By.ID,value = 'roll').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:])
        pitch_v = float(self.driver.find_element(by=By.ID,value = 'pitch').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:])
        xyz_range = float(self.driver.find_element(by=By.ID,value = 'range').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-1:]) 
        xyz_rate = float(self.driver.find_element(by=By.ID,value = 'rate').find_element(by=By.CLASS_NAME,value = 'rate').get_attribute('innerText')[:-3:]) 

        return [x,y,z,xyz_range,xyz_rate,yaw,roll,pitch,yaw_v,roll_v,pitch_v]

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

    def calculate_reward(self,next_state):

        x_reward = (200 - np.abs(next_state[0]))/200
        y_reward = (25 - np.abs(next_state[1]))/25
        z_reward = (25 - np.abs(next_state[2]))/25

        yaw_reward = (20 - np.abs(next_state[5]))/20
        roll_reward = (20 - np.abs(next_state[6]))/20
        pitch_reward = (20 - np.abs(next_state[7]))/20

        
        if next_state[0] < 50 :
            if np.abs(next_state[8]) < 0.2:
                yaw_v_reward = 4
            else:
                yaw_v_reward = -1
            if np.abs(next_state[9]) < 0.2:
                roll_v_reward = 4
            else:
                roll_v_reward = -1
            if np.abs(next_state[10]) < 0.2:
                pitch_v_reward = 4  
            else:
                pitch_v_reward = -1 
            if np.abs(next_state[4]) < 0.2:
                rate_reward = 4  
            else:
                rate_reward = -1                      
        

        elif next_state[0] > 50 and next_state[0] < 100:
            if np.abs(next_state[8]) < 0.4:
                yaw_v_reward = 2
            else:
                yaw_v_reward = -1
            if np.abs(next_state[9]) < 0.4:
                roll_v_reward = 2
            else:
                roll_v_reward = -10
            if np.abs(next_state[10]) < 0.4:
                pitch_v_reward = 2    
            else:
                pitch_v_reward = -1
            if np.abs(next_state[4]) < 1:
                rate_reward = 4  
            else:
                rate_reward = -1 

        else:
            if np.abs(next_state[8]) < 0.5:
                yaw_v_reward = 1
            else:
                yaw_v_reward = -1
            if np.abs(next_state[9]) < 0.5:
                roll_v_reward = 1
            else:
                roll_v_reward = -1
            if np.abs(next_state[10]) < 0.5:
                pitch_v_reward = 1    
            else:
                pitch_v_reward = -1 
            if np.abs(next_state[4]) < 5:
                rate_reward = 1  
            else:
                rate_reward = -1 

        done_reward = 0

        if self.success():
            done_reward = 100        
        
        reward = x_reward+y_reward+z_reward+yaw_reward+roll_reward+pitch_reward+yaw_v_reward+roll_v_reward+pitch_v_reward+rate_reward+done_reward
        return reward

    def step(self,action_number,state):

        done = False
            
        if self.fail() == False and self.success() == False and self.HUD_available():
            self.action(action_number)
            time.sleep(1)
            if self.fail() == False and self.success() == False and self.HUD_available():  # Caso base
                next_state = self.state()
            else:
                next_state = state
                done = True
        else:
            next_state = state
            done = True
        reward = self.calculate_reward(next_state)
        return next_state,reward,done

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