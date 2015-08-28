require '../MyTE_NLP'
local mnist = require 'mnist'

local batch = 20
local A = 28
local seq = 50

local model = rnn.Draw(28 * 28, 100, batch, seq, batch, 28, 28, batch)
   :build(true)
   :apply('enc', 'dec', true)
local params, grads = model:getParameters()
local config = {learningRate = 1e-2}
local trainset = mnist.traindataset()
local testset = mnist.testdataset()

local patch = torch.zeros(batch, 28)
for i = 1, batch do for j = 1, A do patch[i][j] = i end end
local features = torch.zeros(batch, 28, 28)
for i = 1, batch do
   features[{{i}, {}, {}}] = trainset[i].x:gt(125)
end

local feval = function(ps)
   if ps ~= params then params:copy(ps) end
   grads:zero()
   for i = 1, seq do
      model:forward{features, patch}
   end
   local loss = self.loss_t
   loss = loss / seq
   for i = seq, 1, -1 do
      model:backward(patch)
   end
   grads:clamp(-5, 5)
   model:restart()
   return loss, grads
end

for i = 1, 1000 do
   local _, loss = optim.adagrad(feval, params, config)
   if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))      
  end
end
