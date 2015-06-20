require '../MyTE_NLP'

alphabetSize = 71
hiddensize = 100
batchSize = 1
seqSize = 203

local encoder = nn.Sequential()
encoder:add(rnn.Recurrent(alphabetSize, hiddensize, batchSize, true):apply('enc', true))
local decoder = nn.Sequential()
decoder:add(rnn.Recurrent(hiddensize, hiddensize, batchSize, true):apply('dec', true))
decoder:add(nn.Linear(hiddensize, alphabetSize))
decoder:add(nn.Threshold())
decoder:add(nn.LogSoftMax())
local model = rnn.EncDec(encoder, decoder, 'S', alphabetSize, hiddensize, batchSize, seqSize)
local criterion = nn.ClassNLLCriterion():clone(seqSize)
print(model, criterion)

for n = 1, 10 do
   local i = torch.zeros(batchSize, alphabetSize)
   model:forward(i)
   print('#' .. n .. ' in')
end
local o = model:forward('S')
for n = 1, 15 do
   local t = torch.ones(batchSize)
   o = model:decode(o)
   local e = criterion:forward(o, t)
   local d = criterion:backward(o, t)
   model:backward(o, d)
   print('#' .. n .. ' out')
end
--model:forward('S')
