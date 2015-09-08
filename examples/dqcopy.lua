require '../MyTE_NLP'
local cmd = torch.CmdLine()
cmd:option('--agent', 'DeepQ', 'DeepQ | RecurrentDeepQ')
cmd:option('--size', 10, 'n bits')
cmd:option('--layer', 'rec', 'rec | gru')
cmd:option('--nl', 1, '# layers')
cmd:option('--attend', false, 'use attention')
cmd:option('--seq', 4, 'seq length')
cmd:option('--rand', false, 'range vs randperm')
local opt = cmd:parse(arg or {})
print(opt)

local env = { nstates = opt.size, nactions = opt.size }
local aopt = { rnntype = opt.layer, nlayers = opt.nl, attend = opt.attend, seq = opt.seq }
local sm = nn.SoftMax()
local state, counter
local function gen()
   if opt.rand then state = torch.randperm(opt.size)
   else state = torch.range(1, opt.size) end
   counter = 1
end
gen()
local i, correct = 1, 0

local agent = rl[opt.agent](env, aopt)
local s
print(agent)

trainer = rl.RLTrainer(agent,
   function()
      s = state[counter]
      return s
   end,
   function(action)
      local ps = sm:forward(action)
      local max, argmax = ps:max(1)
      argmax = argmax:squeeze()
      local reward = 0

      if s == argmax then
         reward = 1
         correct = correct + 1
      else
         reward = 0
      end
   
      print(i, string.format('%.3f', correct/i),
         s, argmax, reward, trainer.agent.loss)
      i = i + 1

      if counter == opt.size then gen()
      else counter = counter + 1 end

      return reward
   end
)
pcall(trainer.train, trainer)

state = torch.randperm(10)
for i = 1, 10 do
   local si = state[i]
   local action = trainer.agent:forward(si)
   local ps = sm:forward(action)
   local max, argmax = ps:max(1)
   argmax = argmax:squeeze()
   print(si, argmax)
end