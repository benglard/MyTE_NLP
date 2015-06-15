-------- BaseClassifier

local BaseClassifier = torch.class('nlp.BaseClassifier')

function BaseClassifier:__init(data)
   --[[
      REQUIRES:
         data -> a lua table or nil
      EFFECTS:
         Creates an instance of the nlp.BaseClassifier class
         for use in classifiying data.
   ]]

   self.data = {}
   self.labels = nlp.Set()

   if data ~= nil then
      assert(type(data) == 'table',
         'BaseClassifier.__init requires input of type table')
      self.data = data
      self:train(self.data)
   end
end

function BaseClassifier:train(data)
   --[[
      REQUIRES:
         data -> a lua table or nil
      EFFECTS:
         Trains the classifier, here throws a
         NotImplemented error
   ]]

   error('BaseClassifier.train not implemented')
end

function BaseClassifier:classify(input)
   --[[
      REQUIRES:
         input -> a lua table
      EFFECTS:
         Classifies the input, here throws a
         NotImplemented error
   ]]

   error('BaseClassifier.classify not implemented')
end

function BaseClassifier:__tostring()
   --[[
      EFFECTS:
         Returns a string representation a
         nlp.BaseClassifier
   ]]

   return torch.type(self) .. ' with ' .. #self.labels .. ' labels'
end

-------- NaiveBayesClassifier

local NaiveBayesClassifier, _ = torch.class('nlp.NaiveBayesClassifier', 'nlp.BaseClassifier')

function NaiveBayesClassifier:__init(data, estimator)
   --[[
      REQUIRES:
         data -> a lua table or nil
         estimator -> an instance or subclass of nlp.ProbDist or nil
      EFFECTS:
         Creates an instance of the nlp.NaiveBayesClassifier class
         for use in classifiying data using the naive bayes algorithm,
         which is trained to compute P(label|input)
   ]]

   self.estimator = estimator or nlp.ELEProbDist
   BaseClassifier.__init(self, data)
end

function NaiveBayesClassifier:train(data, estimator)
   --[[
      REQUIRES:
         data -> a lua table or nil
         estimator -> an instance or subclass of nlp.ProbDist or nil
      EFFECTS:
         Trains the NaiveBayesClassifier on the data
   ]]

   self.data = data or self.data or {}
   self.estimator = estimator or self.estimator or nlp.ELEProbDist

   local label_fd = nlp.FreqDist()
   local feature_fd = nlp.DefaultTable(nlp.FreqDist)
   local feature_values = nlp.DefaultTable(nlp.Set)
   local fnames = nlp.Set()

   -- Count up how many times each feature value occurred, given
   -- the label and featurename

   for n, elem in pairs(data) do
      local input, label = unpack(elem)
      self.labels:add(label)
      label_fd:push(label)
      for fname, fval in pairs(input) do
         feature_fd{label, fname}:push(fval)
         feature_values(fname):add(fval)
         fnames:add(fname)
      end
   end

   self.label_pd = self.estimator(label_fd)
   self.feature_pd = {}
   for name, fd in pairs(feature_fd.samples) do
      local label, fname = unpack(torch.deserialize(name))
      local pd = self.estimator(fd, #feature_values(fname))
      self.feature_pd[name] = pd
   end
end

function NaiveBayesClassifier:classify(input)
   --[[
      REQUIRES:
         input -> a lua table
      EFFECTS:
         Classifies the input by computing the label
         given the input features
   ]]

   local logprobs = {}   
   for label in self.labels:iter() do
      logprobs[label] = self.label_pd:logprob(label)
   end

   for label in self.labels:iter() do
      for fname, fval in pairs(input) do
         local key = torch.serialize{label, fname}
         local probs = self.feature_pd[key]
         local delta = 0
         if probs ~= nil then
            delta = probs:logprob(fval)
         end
         logprobs[label] = logprobs[label] + delta
      end
   end

   return nlp.TableProbDist(logprobs, true, true)
end

-- Useful metatable functions
NaiveBayesClassifier.__call     = NaiveBayesClassifier.classify
NaiveBayesClassifier.__tostring = BaseClassifier.__tostring