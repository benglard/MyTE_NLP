require '../MyTE_NLP'

local n_input = 4
local n_output = 1
local n_hidden = 25
local batch_size = 1
local seq_length = 5

local model = nn.Sequential()
   :add(nn.Linear(n_input, n_hidden))
   :add(rnn.RecurrentAttention(
      rnn.FFAttention(n_hidden, batch_size, true):apply('att', true),
      rnn.Recurrent(n_hidden, n_hidden, batch_size, true):apply('rnn', true),
      seq_length))
   :add(nn.Linear(n_hidden, n_output))
   :add(nn.LogSoftMax())
local crit = nn.MSECriterion()
print(model)

for i = 1, seq_length do
   local input = torch.rand(batch_size, n_input)
   local output = model:forward(input)
   local target = torch.ones(1)
   local err = crit:forward(output, target)
   local grad = crit:backward(output, target)
   model:backward(input, grad)
   print(err)
end