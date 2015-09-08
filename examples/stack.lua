require '../MyTE_NLP'

local model = nn.Sequential()
   :add(rnn.Stack(4, 20, nil, nil, 2,
      false, false, true):apply('rnnstack', true))
   :add(nn.Linear(20, 1))
   :add(nn.LogSoftMax())
local crit = nn.ClassNLLCriterion()
print(model)

local input = torch.rand(1, 4)
local output = model:forward(input)
local target = torch.ones(1)
local err = crit:forward(output, target)
local grad = crit:backward(output, target)
model:backward(input, grad)
print(model.modules[1].prev_s)

local model = nn.Sequential()
   :add(rnn.Deque(4, 20, nil, nil, 2,
      false, false, true):apply('rnndeque', true))
   :add(nn.Linear(20, 1))
   :add(nn.LogSoftMax())
print(model)

local input = torch.rand(1, 4)
local output = model:forward(input)
local target = torch.ones(1)
local err = crit:forward(output, target)
local grad = crit:backward(output, target)
model:backward(input, grad)
print(model.modules[1].prev_s)