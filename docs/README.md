Here is an example screen from https://beergame.opexanalytics.com/

I will use it to explain state variables in the environment, so you can better understand position of agents and 
order inside state variables. `env` is an instance of class `BeerGame`.

```
env.orders = [[11, 10], [15, 16], [11, 14], [8]]
env.shipments = [[4, 8], [10, 6], [4, 8], [4, 4]]
env.stocks = [-4, -13, -6, 6]
env.turn = 27 
``` 