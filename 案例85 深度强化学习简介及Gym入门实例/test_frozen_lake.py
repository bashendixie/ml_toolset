import gym
env = gym.make("FrozenLake")
env.reset()
for t in range(100):
   print("\nTimestep {}".format(t))
   env.render()
   a = env.action_space.sample()
   ob, r, done, _ = env.step(a)
   if done:
      print("\nEpisode terminated early")
      break