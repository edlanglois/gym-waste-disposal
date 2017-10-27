from gym.envs.registration import register

register(
    id='WasteDisposal-v0',
    entry_point='gym_waste_disposal.envs:WasteDisposalEnv',
    max_episode_steps=1000, )
