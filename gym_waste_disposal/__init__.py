from gym.envs.registration import register

register(
    id='waste-disposal-v0',
    entry_point='gym_waste_disposal.envs:WasteDisposalEnv', )
