import selenium
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


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

        x = float(self.driver.find_element(by=By.ID,value = 'x-range').get_attribute('innerText')[:-2:])
        y = float(self.driver.find_element(by=By.ID,value = 'y-range').get_attribute('innerText')[:-2:])
        z = float(self.driver.find_element(by=By.ID,value = 'z-range').get_attribute('innerText')[:-2:]) 
        
        gameOver = self.driver.find_element(by=By.NAME,value = 'isGameOver')

        yaw = float(self.driver.find_element(by=By.ID,value = 'yaw').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:]) 
        roll = float(self.driver.find_element(by=By.ID,value = 'roll').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])
        pitch = float(self.driver.find_element(by=By.ID,value = 'pitch').find_element(by=By.CLASS_NAME,value = 'error').get_attribute('innerText')[:-1:])
        
        return {"x": x , "y": y , "z": z, "yaw" : yaw, "roll" : roll ,"pitch" : pitch}
        
    def action(self):

        login_button = self.driver.find_element(by=By.ID,value = 'roll-left-button')
        login_button.click()    
        return

    def close(self):
        self.driver.close()