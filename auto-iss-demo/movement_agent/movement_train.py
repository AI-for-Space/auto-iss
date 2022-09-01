from movement_agent import*

agent3 = PPG(save_path = '2M')

agent3.load('1M.pt')

agent3.train(
      num_episodes = 1000000,
      max_steps = 30,
      update_steps = 150,
      num_policy_updates_per_aux = 16,
      save_every = 50000)