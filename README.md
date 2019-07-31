# beer-game-env
Beer Game implemented as an OpenAI Gym environment.

Installation:

1. Create a new conda environment to keep things clean
```
conda create python=3.6 --name beer-game-env
source activate beer-game-env
```

2. Clone the environment repository
```
git clone https://github.com/orlov-ai/beer-game-env
```

3. Point to root repository and install the package
```
cd beer-game-env
pip install -e .
```

To use:
```
import gym
import beer_game_env
env = gym.make('BeerGame-v0', n_agents=4, env_type='classical')
```

tested with gym version `gym==0.14.0`

Need a feature? Have a problem? Just start an issue.
PRs are always welcome.
