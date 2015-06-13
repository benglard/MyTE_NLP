require '../MyTE_NLP'
print(rnn)

local batchSize = 2
local steps = 5

local model = nn.Sequential()
model:add(nn.Linear(200, 100))
model:add(rnn.LSTM(100, batchSize, true):apply('rnn', true))
model:add(nn.Linear(100, 10))
model:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()

model:clone(steps)
criterion:clone(steps)
print(model, criterion)

for i = 1, steps do
   local x = torch.rand(batchSize, 200)
   local label = torch.zeros(batchSize):fill(3)

   local output = model:forward(x)
   local err = criterion:forward(output, label)
   local gradOutput = criterion:backward(output, label)
   model:backward(x, gradOutput)

   print(output)
end

local ps, gs = model:getParameters()
print(#ps)