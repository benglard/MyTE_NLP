# MyTE_RNN

MyTE_RNN is a library built to ease the development and use of recurrent neural network architechtures on torch7 systems. MyTE_RNN seamlessly abstracts nn modules with nngraph gModules, allowing correct and elegant implementations. All of the debugging capabilities afforded by nngraph are built-in to MyTE_RNN.

# Example Usage

```lua
local batchSize = 2
local steps = 5

local model = nn.Sequential()
model:add(nn.Linear(200, 100))
model:add(rnn.LSTM(100, batchSize))
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
```

# Inspirations

MyTE_RNN takes rnn implementation inspiration from these sources:

<p>https://github.com/wojzaremba/lstm</p>
<p>https://github.com/oxford-cs-ml-2015</p>
<p>https://github.com/karpathy/char-rnn</p>
<p>https://github.com/Element-Research/rnn</p>

and it's inspiration on its structure from this source:

https://github.com/koraykv/unsup

