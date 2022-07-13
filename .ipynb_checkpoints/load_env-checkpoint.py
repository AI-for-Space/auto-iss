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

        print('[INFO]Conectando con chrome...')
        chrome_options = Options()
        chrome_options.add_argument('log-level=3')
        chrome_options.add_experimental_option("detach",True)
        driver = webdriver.Chrome(options=chrome_options,service=Service(ChromeDriverManager().install())) 
        driver.get("http://localhost:" + str(localhost) + "/iss-sim.spacex.com")
        self.driver = driver

        # Find login button
        login_button = driver.find_element(by=By.ID,value = 'begin-button')

        # Click login
        while True:
            if login_button.is_displayed():
                text = login_button.get_attribute('innerText')
                login_button.click()
                time.sleep(8)
                break


    def state(self):


        x = float(self.driver.find_element(by=By.ID,value = 'x-range').get_attribute('innerText')[:-1:])
        y = float(self.driver.find_element(by=By.ID,value = 'y-range').get_attribute('innerText')[:-1:])
        z = float(self.driver.find_element(by=By.ID,value = 'z-range').get_attribute('innerText')[:-1:])

        yaw = float(self.driver.find_element(by=By.ID,value = 'yaw').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:]) 
        roll = float(self.driver.find_element(by=By.ID,value = 'roll').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])
        pitch = float(self.driver.find_element(by=By.ID,value = 'pitch').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])

        yaw_v = 0
        roll_v = 0
        pitch_v = 0

        xyz_range = 0
        xyz_rate = 0

        return [x,y,z,xyz_range,xyz_rate,yaw,roll,pitch,yaw_v,roll_v,pitch_v]

    def reset(self):
        restart_button = driver.find_element(by=By.ID,value = 'option-restart')
        restart_button.click()
        return 
        
    def action(self):

        login_button = self.driver.find_element(by=By.ID,value = 'roll-left-button')
        login_button.click()    
        return

    def close(self):
        self.driver.close()