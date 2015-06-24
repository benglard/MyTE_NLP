-- Modeled after https://github.com/bshillingford/autobw.torch/blob/master/examples/rnn_example.lua
require '../MyTE_NLP'

local cmd = torch.CmdLine()
cmd:option('--layer', 'rec', 'rec | lstm | gru')
cmd:option('--debug', false, 'set nngraph debugging mode')
cmd:option('--clone', false, 'clone over time steps')
cmd:option('--kp', false, 'Keep params init from autobw')
local opt = cmd:parse(arg or {})
local layer = nil
if     opt.layer == 'rec'  then layer = rnn.Recurrent
elseif opt.layer == 'lstm' then layer = rnn.LSTM
elseif opt.layer == 'gru'  then layer = rnn.GRU
else error('Invalid layer type') end

local n_input = 1
local n_output = 1
local n_hidden = 25
local batch_size = 15
local seq_length = 5

local model = nn.Sequential()
model:add(layer(n_input, n_hidden, batch_size, true):apply('rnn', opt.debug))
model:add(nn.Linear(n_hidden, n_output))
local criterion = nn.MSECriterion()

if opt.clone then
   model:clone(seq_length)
   criterion:clone(seq_length)
end

local data = torch.linspace(0, 20*math.pi, 1000):sin():view(-1, 1)
local start_idx = torch.Tensor(batch_size):uniform():mul(data:size(1) - seq_length):ceil():long()
local batch = torch.zeros(seq_length, batch_size, 1)

local function next_batch()
   start_idx:add(-1)
   for i = 1, seq_length do
      start_idx:apply(function(x) return x % data:size(1) + 1 end)
      batch:select(1, i):copy(data:index(1, start_idx):view(1, -1, 1))
   end
   return batch:clone()
end

local params, grads = model:getParameters()
if opt.kp then params:uniform(-0.1, 0.1) end

local function fopt(x)
   if params ~= x then
      params:copy(x)
   end
   grads:zero()

   local batch = next_batch()
   local inputs = batch:sub(1, batch:size(1)-1)
   local targets = batch:sub(2, batch:size(1))

   local loss = 0
   for t = 1, inputs:size(1) do
      local output = model:forward(inputs[t])
      local err = criterion:forward(output, targets[t])
      local gradOutput = criterion:backward(output, targets[t])
      model:backward(inputs[t], gradOutput)
      loss = loss + err
   end

   return loss, grads
end

local s = 0
for i = 1, 10000 do
   local _, fx = optim.sgd(fopt, params, {})
   print(i, fx[1])
   s = s + fx[1]
end
print(s)