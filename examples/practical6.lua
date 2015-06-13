require '../MyTE_NLP'
local CharLMMinibatchLoader = require './CharLMMinibatchLoader'
include 'Embedding.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-vocabfile','./data/vocab.t7','filename of the string->int table')
cmd:option('-datafile','./data/train.t7','filename of the serialized torch ByteTensor to load')
cmd:option('-batch_size',16,'number of sequences to train on in parallel')
cmd:option('-seq_length',16,'number of timesteps to unroll to')
cmd:option('-rnn_size',256,'size of LSTM internal state')
cmd:option('-max_epochs',1,'number of full passes through the training data')
cmd:option('-savefile','./models/practical6.model','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()
opt = cmd:parse(arg)
print(opt)
torch.manualSeed(opt.seed)

local loader = CharLMMinibatchLoader.create(opt.datafile, opt.vocabfile, opt.batch_size, opt.seq_length)
local vocab_size = loader.vocab_size  -- the number of distinct characters

local model = nn.Sequential()
model:add(nn.LookupTable(vocab_size, opt.rnn_size))
model:add(nn.Reshape(opt.batch_size * opt.seq_length, opt.rnn_size))
model:add(rnn.LSTM(opt.rnn_size, opt.batch_size, true):apply('lstm', true))
model:add(nn.Linear(opt.rnn_size, vocab_size))
model:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()
model:clone(opt.seq_length)
criterion:clone(opt.seq_length)

local params, grad_params = model:getParameters()
params:uniform(-0.08, 0.08)

function feval(x)
   if x ~= params then params:copy(x) end
   grad_params:zero()

   local x, y = loader:next_batch()
   local output = model:forward(x)
   local err = criterion:forward(output, y)
   local gradOutput = criterion:backward(output, y)
   model:backward(x, gradOutput)
   grad_params:clamp(-5, 5)

   return loss, grad_params
end

local losses = {}
local optim_state = {learningRate = 1e-1}
local iterations = opt.max_epochs * loader.nbatches
for i = 1, iterations do
   local _, loss = optim.adagrad(feval, params, optim_state)
   losses[#losses + 1] = loss[1]

   if i % opt.print_every == 0 then
      print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e", i, loss[1], grad_params:norm()))
   end
end