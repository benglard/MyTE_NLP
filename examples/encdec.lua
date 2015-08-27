require '../MyTE_NLP'

alphabetSize = 71
hiddensize = 100
batchSize = 1
seqSize = 20

local encoder = nn.Sequential()
encoder:add(rnn.Recurrent(alphabetSize, hiddensize, batchSize, true):apply('enc', true))
local decoder = nn.Sequential()
decoder:add(rnn.Recurrent(hiddensize, hiddensize, batchSize, true):apply('dec', true))
decoder:add(nn.Linear(hiddensize, alphabetSize))
decoder:add(nn.Threshold())
decoder:add(nn.LogSoftMax())
local model = rnn.EncDec(encoder, decoder, 'S', alphabetSize, hiddensize, batchSize, seqSize, alphabetSize)
local criterion = nn.ClassNLLCriterion():clone(seqSize)
print(model, criterion)

model:training()
for n = 1, 10 do
   local i = torch.zeros(batchSize, alphabetSize)
   model:forward(i)
   print('#' .. n .. ' in')
end
local o = model:forward('S'):clone()
for n = 1, 15 do
   local t = torch.ones(batchSize)
   local lsm = model:forward(o):clone()
   local e = criterion:forward(lsm, t)
   local d = criterion:backward(lsm, t)
   model:backward(o, d)
   o = model:state()
   print('#' .. n .. ' out')
end
model:forward('S')
