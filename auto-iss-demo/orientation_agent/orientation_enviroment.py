import numpy as np

class orientation_enviroment():

    def __init__(self):
        return
    
    def reset(self):
        self.state = [float(np.random.randint(1,4)),float(np.random.randint(1,4)),float(np.random.randint(1,4))]  
        return

    def test_state(self):
        self.state = [float(np.random.randint(1,4)),float(np.random.randint(1,4)),float(np.random.randint(1,4))]  
        return   


    def test_step(self,action_number): 
        
        if self.state[action_number] > 0:
            self.state[action_number] -= 1
        else:
            self.state[action_number] += 1

        if self.state[0] == 0 and self.state[1] == 0 and self.state[2] == 0:
            return self.state,True
        else:
            return self.state,False


    def train_step(self,action_number): 
    
        self.state[action_number] -= 1

        a = self.state[0]
        b = self.state[1]
        c = self.state[2]

        if a == 0 and b == 0 and c == 0:
            return self.state,999999999,True
        if a < 0 or b < 0 or c < 0:
            return self.state,-1,True
        
        reward1 = 27 - ((a*a) + (b*b) + (c*c))

        if a == 0:
            reward1 += 10
        if b == 0:
            reward1 += 10
        if c == 0:
            reward1 += 10 
 
        return self.state,reward1,False


