-- Modeled after http://arxiv.org/pdf/1503.01007v4.pdf task 1
require '../MyTE_NLP'

local cmd = torch.CmdLine()
cmd:option('--nstacks', 1, 'number of stacks')
cmd:option('--verbose', false, 'print useful stuff')
cmd:option('--rnn', false, 'use plain rnns')
cmd:option('--deque', false, 'use rnn.Deque')
cmd:option('--layer', 'rec', 'rec | lstm | gru')
cmd:option('--debug', false, 'set nngraph debugging mode')
cmd:option('--clone', false, 'clone over time steps')
cmd:option('--kp', false, 'Keep params init from autobw')
cmd:option('--n', 10000, 'n iterations')
cmd:option('--exp', 1, 'exp of a^expb^exp sequence')
cmd:option('--discrete', false, 'discretize stacks')
cmd:option('--noop', false, 'use noop')
cmd:option('--endon', false, 'end on batch_size')
local opt = cmd:parse(arg or {})
print(opt)

local n_input = 1
local n_output = 2
local n_hidden = 40
local batch_size = 15
local seq_length = 5

local model = nn.Sequential()

if opt.rnn then
   local layer = nil
   if     opt.layer == 'rec'  then layer = rnn.Recurrent
   elseif opt.layer == 'lstm' then layer = rnn.LSTM
   elseif opt.layer == 'gru'  then layer = rnn.GRU
   else error('Invalid layer type') end

   model:add(layer(n_input, n_hidden, 1, true):apply('rnn1', opt.debug))
   for i = 2, opt.nstacks do
      local name = string.format('rnn%d', i)
      model:add(layer(n_hidden, n_hidden, 1, true):apply(name, opt.debug))
   end
else
   local layer = nil
   if opt.deque then layer = rnn.Deque
   else layer = rnn.Stack end
   model:add(layer(n_input, n_hidden, n_hidden, 2, opt.nstacks,
      opt.discrete, opt.noop, true):apply('stack1', opt.debug))
end

model:add(nn.Linear(n_hidden, n_output))
model:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()

if opt.clone then
   model:clone(seq_length)
   criterion:clone(seq_length)
end
print(model)

local onA = true
local cc = 1
local function next_batch()
   local rv
   if cc == opt.exp then
      if onA then rv = {1, 2} else rv = {2, 1} end
      cc = 1
      onA = not onA
   else
      cc = cc + 1
      if onA then rv = {1, 1} else rv = {2, 2} end
   end
   return rv
end

local params, grads = model:getParameters()
if opt.kp then params:uniform(-0.1, 0.1) end

local function fopt(x)
   if params ~= x then
      params:copy(x)
   end
   grads:zero()

   local loss = 0
   local correct = 0

   for i = 1, batch_size do
      local i, t = unpack(next_batch())
      local input = torch.zeros(1, 1):add(i)
      local target = torch.zeros(1):add(t)
      local output = model:forward(input)

      local m, am = torch.exp(output[1]):max(1)
      am = am:squeeze()
      if am == t then correct = correct + 1 end
      if opt.verbose then
         local temp = 'I: %d, T: %d, O: %d'
         local str = string.format(temp, i, t, am)
         print(str)
      end

      local err = criterion:forward(output, target)
      local gradOutput = criterion:backward(output, target)
      model:backward(input, gradOutput)
      loss = loss + err
   end

   print(correct, correct == batch_size)
   if correct == batch_size and opt.endon then os.exit() end
   return loss, grads
end

local s = 0
for i = 1, opt.n do
   local _, fx = optim.sgd(fopt, params, {})
   print(i, fx[1])
   s = s + fx[1]
end
print(s)