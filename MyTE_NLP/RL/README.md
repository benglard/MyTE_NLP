# MyTE_RL

MyTE_RL is a library built to ease the development and use of reinforcement learning architechtures on torch7 systems. MyTE_RL is built on top of nn, and MyTE_RNN, allowing for the building of powerful models.

# Example Usage

This is a toy example where an agent can exist in one of two states, and it's goal is to move from state to state.

```lua
local s = 1
local sm = nn.SoftMax()
rl.RLTrainer(
   rl.DeepQ{ nstates = 2, nactions = 2 },
   function()
      local state = torch.zeros(2)
      state[s] = 1.0
      if s == 1 then s = 2 else s = 1 end
      return state
   end,
   function(action)
      local ps = sm:forward(action)
      local p1 = ps[1]
      local p2 = ps[2]
      if s == 1 and (p2 > p1) then return 1 end
      if s == 2 and (p1 > p2) then return 1 end
      return 0
   end
):train{verbose=true}
```