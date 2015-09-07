require '../MyTE_NLP'
local cmd = torch.CmdLine()
cmd:option('--layer', 'rec', 'rec | gru')
cmd:option('--nl', 1, '# layers')
local opt = cmd:parse(arg or {})
local env = { nstates = 2, nactions = 2 }
local aopt = { rnntype = opt.layer, nlayers = opt.nl }

local agent = rl.RecurrentDeepQ(env, aopt)
print(agent)
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
   print(i, string.format('%.3f %.3f %.3f', correct/i, p1, p2),
      state, nstate, reward, agent.loss)

   i = i + 1
   if state == 1 then state = 2
   else state = 1
   end
end