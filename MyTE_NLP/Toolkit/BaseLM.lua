-------- BaseLM

local BaseLM = torch.class('nlp.BaseLM')

function BaseLM:__init(documents)
   --[[
      REQUIRES:
         documents -> a lua table or nlp.String or lua string
      EFFECTS:
         Creates an instance of the nlp.BaseLM class, a baseclass
         for use in building language models.
   ]]

   self.documents = {}

   local t = torch.type(documents)
   if t == 'table' then
      for n, item in pairs(documents) do
         self.documents[n] = item
      end
   elseif t == 'nlp.String' or t == 'string' then
      self.documents = documents:split(' ')
   else
      error('BaseLM.__init requires argument of type table or nlp.String')
   end

   if #self.documents > 0 then
      self.first = self.documents[1]
   else
      self.first = ''
   end
end

function BaseLM:sample(context, nsamples)
   --[[
      REQUIRES:
         context -> a nlp.String or string
         nsamples -> a number
      EFFECTS:
         Samples nsamples times given context
         from the language model, here throws
         a NotImplemented error
   ]]

   error('BaseLM.sample not implemented')
end

function BaseLM:__len()
   --[[
      EFFECTS:
         Returns the length of a nlp.BaseLM,
         given by the number of documents
   ]]

   return #self.documents
end

function BaseLM:__tostring()
   --[[
      EFFECTS:
         Returns a string representation
         of a nlp.BaseLM
   ]]

   return torch.type(self) .. ' with ' .. #self .. ' documents'
end

-------- NgramLM

local NgramLM, _ = torch.class('nlp.NgramLM', 'nlp.BaseLM')

function NgramLM:__init(documents, n, estimator, pad_left, pad_right)
   --[[
      REQUIRES:
         documents -> a lua table or nlp.String or lua string
         n -> a number, n in ngram
         estimator -> a probability distribution or factory
            function producing a probability distribution
         pad_left -> if true, pads the documents at the start
         pad_right -> if true, pads the documents at the end
      EFFECTS:
         Creates an instance of the nlp.NgramLM class for use
         in modeling and sampling from a Katz ngram backoff model.
         http://en.wikipedia.org/wiki/Katz's_back-off_model
   ]]

   BaseLM.__init(self, documents)
   self.n = n or 2
   self.estimator = estimator
   if pad_left == nil then pad_left = false end
   self.pad_left = pad_left
   self.pad_right = pad_right or false

   -- Create paddings
   self.lpad = {}
   if pad_left then
      for i = 1, n - 1 do
         self.lpad[i] = ''
      end
   end
   self.rpad = {}
   if pad_right then
      for i = 1, n - 1 do
         self.rpad[i] = ''
      end
   end

   cfd = nlp.ConditionalFreqDist()
   self.ngrams = nlp.Set()
   local grams = nlp.ngrams(self.documents, self.n,
      self.pad_left, self.pad_right, '')
   for n, ngram in pairs(grams) do
      self.ngrams:add(ngram)
      local context = {}
      for i = 1, self.n - 1 do
         context[i] = ngram[i]
      end
      local token = ngram[self.n]
      cfd:push(context, token)
   end

   self.model = nlp.ConditionalProbDist(cfd, self.estimator, #cfd)

   -- recursively construct the lower-order models
   if self.n > 1 then
      self.backoff = nlp.NgramLM(self.documents, self.n - 1,
         self.estimator, self.pad_left, self.pad_right)
   end
end

function NgramLM:prob(context, token)
   --[[
      REQUIRES:
         context -> a lua table or nlp.String or string
         token -> a string
      EFFECTS:
         Calculates the probability of token
         appearing after context
   ]]

   local t = torch.type(context)
   local previous = {}
   if t == 'string' or t == 'nlp.String' then
      previous = context:split(' ')
   elseif t == 'table' then
      previous = context
   else
      error('NgramLM.prob requires argument of type table or nlp.String')
   end
 
   local ngram = table.deepcopy(previous)
   ngram[self.n] = token

   if self.ngrams:contains(ngram) then
      return self.model:get(ngram):prob(token)
   elseif self.n == 1 then
      return self.model:get(previous):prob(token)
   else
      table.remove(ngram, self.n)
      table.remove(ngram, 1)
      return self:alpha(previous) * self.backoff:prob(ngram, token)
   end
end

function NgramLM:alpha(context)
   --[[
      REQUIRES:
         context -> a lua table
      EFFECTS:
         Calculates the alpha paramater for Katz Backoff model
   ]]

   local tokens = table.deepcopy(context)
   table.remove(tokens, 1)
   return self:beta(context) / self.backoff:beta(tokens)
end

function NgramLM:beta(context)
   --[[
      REQUIRES:
         context -> a lua table
      EFFECTS:
         Calculates the beta paramater for Katz Backoff model
         which represents the left-over probability mass for the
         (n-1)-gram
   ]]

   local pd = self.model:get(context)
   if pd == nil then
      return 1
   else
      return pd:discount()
   end
end
  
function NgramLM:sample(context, nsamples)
   --[[
      REQUIRES:
         context -> a nlp.String or string
         nsamples -> a number
      EFFECTS:
         Samples nsamples times given context
         from the language model
   ]]

   local text = {}
   local t = torch.type(context)
   if t == 'string' or t == 'nlp.String' then
      text = context:split(' ')
   elseif t == 'table' then
      text = context
   else
      error('NgramLM.sample requires a first argument of type table or nlp.String')
   end

   assert(type(nsamples) == 'number',
      'NgramLM.samples requires a second argument of type number')
   for i = 1, nsamples do
      table.insert(text, self:_sample_one(text))
   end
   return table.concat(text, ' ')
end

function NgramLM:_sample_one(context)
   --[[
      REQUIRES:
         context -> a nlp.String or string
      EFFECTS:
         Samples 1 times given context
         from the language model
   ]]

   local previous = {}
   table.deepcopy(self.lpad, previous)
   table.deepcopy(context, previous)

   local stop = #previous
   local start = stop - self.n + 2
   local ngram = {}
   context = {}
   for i = start, stop do
      local prev = previous[i]
      table.insert(context, prev)
   end

   local pd = self.model:get(ngram)
   if pd ~= nil then
      return pd:generate()
   elseif self.n > 1 then
      local next_c = table.deepcopy(context)
      table.remove(next_c, 1)
      return self.backoff:_sample_one(next_c)
   else
      return '.'
   end
end

function NgramLM:get(key)
   --[[
      REQUIRES:
         key -> a string or nlp.String or typical lua key
      EFFECTS:
         Gets the probability distribution associated
         with a given key (context) 
   ]]

   local modelkey = {}
   local t = torch.type(key)

   if t == 'string' or t == 'nlp.String' then
      modelkey = key:split(' ')
   else
      modelkey = key
   end

   return self.model:get(modelkey)
end

function NgramLM:__tostring()
   --[[
      EFFECTS:
         Returns a string representation
         of a nlp.NgramLM
   ]]

   return torch.type(self) .. ' with ' .. #self .. ' documents of size ' .. self.n
end

-- Useful metatable functions
NgramLM.__len  = BaseLM.__len
NgramLM.__call = NgramLM.get
