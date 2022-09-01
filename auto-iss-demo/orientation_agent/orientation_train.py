from orientation_agent import*

agent0 = PPG(save_path = '1M')

agent0.train(
      num_episodes = 1000000,
      max_steps = 30,
      update_steps = 150,
      num_policy_updates_per_aux = 16,
      save_every = 50000)

