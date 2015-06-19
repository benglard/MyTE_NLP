require '../MyTE_NLP'

alphabetSize = 71
hiddensize = 100
batchSize = 1
seqSize = 203

local encoder = nn.Sequential()
encoder:add(rnn.Recurrent(alphabetSize, hiddensize, batchSize, true):apply('enc', true))
local decoder = nn.Sequential()
decoder:add(rnn.Recurrent(alphabetSize, hiddensize, batchSize, true):apply('dec', true))
decoder:add(nn.Linear(hiddensize, alphabetSize))
decoder:add(nn.Threshold())
decoder:add(nn.LogSoftMax())
local model = rnn.EncDec(encoder, decoder, alphabetSize, seqSize, 'S')
local criterion = nn.ClassNLLCriterion():clone(seqSize)
print(model, criterion)

for i = 1, 10 do
   local i = torch.zeros(batchSize, alphabetSize)
   model:forward(i)
end
model:forward('S')
for i = 1, 5 do
   local i = torch.zeros(batchSize, alphabetSize)
   local t = torch.ones(batchSize)
   local o = model:forward(i)
   local e = criterion:forward(o, t)
   local d = criterion:backward(o, t)
   model:backward(i, d)
end
--model:forward('S')
