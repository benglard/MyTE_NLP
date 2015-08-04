require '../MyTE_NLP'

local vocab = { 'hi', 'hello', 'bye', 'goodbye', '<eos>' }
local rvcoab = {
   hi = 1,
   hello = 2,
   bye = 3,
   goodbye = 4,
   ['<eos>'] = 5
}

local batchsize = 1
local seqsize = 4
local vocabsize = 5
local hiddensize = 20
local stop = '<eos>'

local encoder = nn.Sequential()
encoder:add(nn.LookupTable(vocabsize, hiddensize))
encoder:add(rnn.Recurrent(hiddensize, hiddensize, batchsize))

local env = { nstates = hiddensize, nactions = 2 }
local deepq = rl.DeepQ(env)

local model = rl.DQEncDec(encoder, deepq,
   stop, 1, hiddensize, batchsize, seqsize, vocabsize)

print(model)
model:training()

local s_1 = torch.zeros(1):fill(1)
local s_2 = torch.zeros(1):fill(2)
local state = 1 -- 'hi' before 'hello'
local sm = nn.SoftMax()
local i = 1
local correct = 0
local config = {
   learningRate = 1e-2,
   learningRateDecay = 5e-7,
   momentum = 0.9
}

while true do
   if state == 1 then
      model:forward(s_1:clone())
      model:forward(s_2:clone())
   else
      model:forward(s_2:clone())
      model:forward(s_1:clone())
   end

   local output = model:forward('<eos>'):clone():resize(hiddensize)
   local action = model:forward(output):clone()
   
   local probs = sm:forward(action)
   local p1 = probs[1]
   local p2 = probs[2]
   local reward = 0
   local _, pred = probs:max(1)
   pred = pred:squeeze()

   if state == 1 then
      if p1 > p2 then
         reward = 1
         correct = correct + 1
      end
   else
      if p2 > p1 then
         reward = 1
         correct = correct + 1
      end
   end

   model:backward(reward, config)
   model:forward('<eos>')

   print(i, string.format('%.3f %.3f %.3f', correct/i, p1, p2),
      state, pred, reward, model.decoder.loss)
   i = i + 1
   if state == 1 then state = 2
   else state = 1 end
   model:restart()
end