from DEMO import *

orientation_agent = PPG()
movement_agent = PPG()
# Cargar modelo
orientation_agent.load('orientation_net.pt')
movement_agent.load('movement_net.pt')

play_demo(orientation_agent,movement_agent)