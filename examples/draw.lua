require '../MyTE_NLP'
require 'image'
local mnist = require 'mnist'

local cmd = torch.CmdLine()
cmd:option('--verbose', false, 'verbose')
cmd:option('--debug', false, 'debug')
cmd:option('--n', 1000, 'n epochs')
local opt = cmd:parse(arg or {})
print(opt)

local batch = 20
local A = 28
local seq = 50
local hidden = 100

local model = rnn.Draw(A * A, hidden, batch, seq, batch, A, A, 3, opt.verbose)
   :build(true)
   :apply('enc', 'dec', opt.debug)
print(model)
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
   local loss = model.loss_t
   loss = loss / seq
   for i = seq, 1, -1 do
      model:backward(patch)
   end
   grads:clamp(-5, 5)
   model:restart()
   return loss, grads
end

for i = 1, opt.n do
   local _, loss = optim.adagrad(feval, params, config)
   if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))      
  end
end

local path = os.getenv('HOME') .. '/Desktop/draw'
model:generate(features, patch, true, path)