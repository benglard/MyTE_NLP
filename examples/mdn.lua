require '../MyTE_NLP'

local n_gaussians = 2
local n_features = 5
local n_labels = 2

local model = rnn.MDN(n_features, n_gaussians, n_labels, true)
local criterion = nn.MSECriterion()
local params, grads = model:getParameters()
local config = {learningRate = 1e-1}

local n_data = 100
local features_input = torch.zeros(n_data, n_features)
local labels_input = torch.zeros(n_data, n_labels)
for i = 1, n_data do 
   local element = torch.ones(n_features)
   element:mul(i)
   element[2] = 0.5
   element[3] = 1
   element:add(torch.randn(element:size()))
   element:div(n_data)
   features_input[{{i}, {}}] = element

   local label = torch.ones(n_labels)
   label[2] =  i
   label[1] = i ^ 2
   label:add(torch.randn(label:size()))
   label:div(n_data ^ 2)
   labels_input[{{i}, {}}] = label  
end

local feval = function(x)
   if x ~= params then params:copy(x) end
   grads:zero()
   local output = model:forward{features_input, labels_input}
   local loss = torch.mean(output)
   model:backward()
   return loss, grads
end

for i = 1, 10000 do
   local _, loss = optim.adagrad(feval, params, config)
   if i % 10 == 0 then
      print(string.format('iteration %4d, loss = %6.6f', i, loss[1]))
   end
end