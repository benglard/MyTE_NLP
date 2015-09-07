require '../MyTE_NLP'
local cmd = torch.CmdLine()
cmd:option('--algo', 'copy', 'copy | sort')
cmd:option('--size', 2, 'n bits')
cmd:option('--verbose', false, 'verbose')
local opt = cmd:parse(arg or {})
print(opt)

local env = { nstates = 2*(opt.size), nactions = opt.size + 1 }
local dqopt = {}
local params = { verbose = opt.verbose }
local sm = nn.SoftMax()
local state, prediction, counter
local i, correct = 1, 0

local function start()
   --state = torch.rand(opt.size + 1):round():add(1)
   --state[opt.size + 1] = opt.size + 1
   state = torch.Tensor{1,2,0,0}
   prediction = {}
   counter = 1
   --print(state)
end

start()

trainer = rl.RLTrainer(
   rl.DeepQ(env, dqopt),
   function() return state end,
   function(action)
      local ps = sm:forward(action)
      local max, argmax = ps:max(1)
      argmax = argmax:squeeze()
      local reward = 0

      --[[local ok, si = pcall(function() return state[counter] end)
      if ok and si == argmax then
         reward = 1
         correct = correct + 1
      else
         reward = -1
      end]]

      print('here', state, counter, argmax)

      if argmax == opt.size + 1 then -- commit
         for i = 1, opt.size do
            local si = state[i]
            local pi = state[i + 2]
            if si == pi then
               reward = reward + 1
            end
         end
      else -- predict
         state[counter + 2] = argmax
      end

      print('here2', state, reward)
      print(trainer.agent.loss)

      if counter == opt.size then
         start()
      else
         counter = counter + 1
      end
      return reward
   end
)
trainer:train(params)