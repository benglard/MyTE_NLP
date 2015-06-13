collectgarbage()
require '../MyTE_NLP'

local cmd = torch.CmdLine()
cmd:option('--agent', 'DeepQ', 'DeepQ | Reinforce | RecurrentReinforce')
cmd:option('--rnn', 'rnn', 'rnn | gru')
cmd:option('--seed', 1, 'random seed')
local opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)

local env = { nstates = 2, nactions = 2 }
local opts = { rnn = opt.rnn, momentum = 0 }
local rnnops = { debug = true, name = 'rnn', args = { 2, 100, 1, true } }

local agent = rl[opt.agent](env, opts, rnnops)
local i = 1
local state = 1
local correct = 0
local sm = nn.SoftMax()

while true do
   local s = torch.zeros(2)
   s[state] = 1.0
   local action = agent:forward(s)
   local reward = 0
   local probs = sm:forward(action)
   local p1 = probs[1]
   local p2 = probs[2]
   local nstate = 0
   if state == 1 then
      if p2 > p1 then
         reward = 1
         correct = correct + 1
         nstate = 2
      else
         nstate = 1
      end
   else
      if p1 > p2 then
         reward = 1
         correct = correct + 1
         nstate = 1
      else
         nstate = 2
      end
   end
   agent:backward(reward)
   print(i, string.format('%.3f %.3f %.3f', correct/i, p1, p2), state, nstate, reward, agent.loss)

   i = i + 1
   if state == 1 then
      state = 2
   else
      state = 1
   end
end

--[[local s = 1
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
):train{verbose=true}]]