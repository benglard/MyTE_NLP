require '../MyTE_NLP'

local data = {}
local ntrain = 100

for i = 1,  ntrain + 1 do
   local x = {
      math.random() + 5,
      math.random() + 5
   }
   local y = 1
   table.insert(data, {x, y})
end

for i = 1,  ntrain do
   local x = {
      math.random() - 5,
      math.random() - 5
   }
   local y = 0
   table.insert(data, {x, y})
end

torch.shuffle(data)
local model = nlp.NaiveBayesClassifier(data)
print(model)
print(model:classify{5, 5}:max())