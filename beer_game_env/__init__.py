from gym.envs.registration import register

register(
    id='BeerGame-v0',
    entry_point='beer_game_env.envs:BeerGame',
)