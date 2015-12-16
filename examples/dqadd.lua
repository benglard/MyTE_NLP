require '../MyTE_NLP'
local cmd = torch.CmdLine()
cmd:option('--agent', 'DeepQ', 'DeepQ | RecurrentDeepQ | DeepQStack')
cmd:option('--size', 10, 'n bits')
cmd:option('--layer', 'rec', 'rec | gru')
cmd:option('--nl', 1, '# layers')
cmd:option('--attend', false, 'use attention')
cmd:option('--seq', 4, 'seq length')
cmd:option('--deque', false, 'use rnn deque')
local opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(os.time())

local env = { nstates = 2 * opt.size, nactions = 2 }
local aopt = { rnntype = opt.layer, nlayers = opt.nl,
   attend = opt.attend, seq = opt.seq, deque = opt.deque }
local sm = nn.SoftMax()
local state, endstate
local function gen()
   state = torch.rand(1):mul(opt.size):int():add(1):squeeze()
   endstate = torch.rand(1):mul(opt.size):int():add(1):squeeze()
end
gen()
local i, correct = 1, 0

local agent = rl[opt.agent](env, aopt)
local s
print(agent)

trainer = rl.RLTrainer(
   agent,
   function() return state end,
   function(action)
      local ps = sm:forward(action)
      local max, argmax = ps:max(1)
      argmax = argmax:squeeze()
      local reward = 0

      if endstate == state then
         reward = 1
         correct = correct + 1
         gen()
      else
         if argmax == 1 then
            state = state + 1
         else
            state = state - 1
         end
         if state < 1 or state > opt.size * 2 then
            reward = -10
            gen()
         else
            reward = -1
         end
      end
   
      print(i, string.format('%.3f', correct/i),
         endstate, state, argmax,
         reward, trainer.agent.loss)
      i = i + 1
      return reward
   end
)
trainer:train()