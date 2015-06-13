local ninf = math.log(0, 2) -- negative infinity

-------- ProbDist

local ProbDist = torch.class('nlp.ProbDist')

-- true if this probability distribution needs to sum to one
ProbDist.SUM_TO_ONE = true

function ProbDist:__init()
   --[[
      EFFECTS:
         Creates an instance of the nlp.ProbDist class,
         a base class for modeling probability
         distributions.
   ]]

   self.samples = {}
   self.N = 0
end

function ProbDist:prob(name)
   --[[
      REQUIRES:
         name -> the name of a sample
         in the distribution
      EFFECTS:
         Returns the probability of name,
         here represented as the value of
         self.samples[name] or 0
   ]]

   return self.samples[name] or 0
end

function ProbDist:logprob(name)
   --[[
      REQUIRES:
         name -> the name of a sample
         in the distribution
      EFFECTS:
         Returns the base 2 log of the
         probability of name
   ]]

   local p = self:prob(sample)
   return math.log(p, 2)
end

function ProbDist:keys()
   --[[
      EFFECTS:
         Returns the sample names as
         a lua table
   ]]

   local rv = {}
   for k, v in pairs(self.samples) do
      table.insert(rv, k)
   end
   return rv
end

function ProbDist:values()
   --[[
      EFFECTS:
         Returns the sample value as
         a lua table
   ]]

   local rv = {}
   for k, v in pairs(self.samples) do
      table.insert(rv, v)
   end
   return rv
end

function ProbDist:generate()
   --[[
      EFFECTS:
         Samples from the given probability
         distribution and returns the 
         associated sample name.
   ]]

   local keys = self:keys()
   local values = self:values()
   local probs = torch.Tensor(values)
   local samples = torch.multinomial(probs, 1)
   return keys[samples[1]]
end

function ProbDist:discount()
   --[[
      EFFECTS:
         Returns the ratio by which counts
         are discounted on average.
   ]]

   return 0
end

function ProbDist:max()
   --[[
      EFFECTS:
         Returns the sample with maximum
         probability. Here, it throws a
         NotImplementedError
   ]]

   error('ProbDist.max not implemented')
end

function ProbDist:__len()
   --[[
      EFFECTS:
         Returns the length of the given
         probability distribution, here
         represented as the number of samples
   ]]

   local n = 0
   for k, v in pairs(self.samples) do n = n + 1 end
   return n
end

function ProbDist:__tostring()
   --[[
      EFFECTS:
         Returns a string representation 
         of the given probability distribution
   ]]

   return torch.type(self) .. ' with ' .. #self .. ' keys storing ' .. self.N .. ' values'
end

-------- UniformProbDist

local UniformProbDist, _ = torch.class('nlp.UniformProbDist', 'nlp.ProbDist')

function UniformProbDist:__init(samples)
   --[[
      REQUIRES:
         samples -> a lua table
      EFFECTS:
         Creates an instance of the nlp.UniformProbDist
         class, for use in modeling a probability
         distribution with equal probability given to every
         member.
   ]]

   assert(type(samples) == 'table',
      'UniformProbDist.__init requires input of type table')
   ProbDist.__init(self)
   self.N = #samples
   for i, elem in pairs(samples) do
      self.samples[elem] = 1.0 / self.N
   end
end

function UniformProbDist:max()
   --[[
      EFFECTS:
         Returns the first sample
   ]]

   local key, _ = next(self.samples)
   return key
end

-- Useful metatable functions
UniformProbDist.__call     = UniformProbDist.prob
UniformProbDist.__tostring = ProbDist.__tostring
UniformProbDist.__len      = ProbDist.__len

-------- TableProbDist

local TableProbDist, _ = torch.class('nlp.TableProbDist', 'nlp.ProbDist')

function TableProbDist:__init(samples, log, normalize)
   --[[
      REQUIRES:
         samples -> a lua table or nil
         log -> if true, logs all the sample probabilities
         normalize -> if true, normalizes all the sample probabilities
      EFFECTS:
         Creates an instance of the nlp.TableProbDist class
         for use in modeling probability distributions given
         by a lua table with keys equaling the sample names
         and values equaling the probabilties.
   ]]

   ProbDist.__init(self)
   samples = samples or {}
   log = log or false
   normalize = normalize or false

   self.N = 0
   self.log = log
   local sum = 0.0
   
   for k, v in pairs(samples) do
      if log then v = math.log(v, 2) end
      self.samples[k] = v
      sum = sum + v
      self.N = self.N + 1
   end

   if normalize then
      for k, v in pairs(self.samples) do
         self.samples[k] = v / sum
      end
   end
end

function TableProbDist:prob(name)
   --[[
      REQUIRES:
         name -> the name of a sample
      EFFECTS:
         Computes the probability of a given sample
   ]]

   if self.log then
      local v = self.samples[name]
      if v == nil then return 0
      else return math.pow(v, 2) end
   else
      return self.samples[name] or 0
   end
end

function TableProbDist:logprob(name)
   --[[
      REQUIRES:
         name -> the name of a sample
      EFFECTS:
         Computes the log-probability of a given sample
   ]]

   if self.log then
      return self.samples[name] or ninf
   else
      local v = self.samples[name]
      if v == nil then return ninf
      else return math.log(v, 2) end
   end
end

function TableProbDist:max()
   --[[
      EFFECTS:
         Computes the sample with
         maximum probability
   ]]

   local name, _ = next(self.samples)
   local max = self:prob(name)
   for key, val in pairs(self.samples) do
      local p = self:prob(key)
      if p > max then
         max = p
         name = key
      end
   end
   return name
end

-- Useful metatable functions
TableProbDist.__call     = TableProbDist.prob
TableProbDist.__tostring = ProbDist.__tostring
TableProbDist.__len      = ProbDist.__len

-------- MLEProbDist

local MLEProbDist, _ = torch.class('nlp.MLEProbDist', 'nlp.ProbDist')

function MLEProbDist:__init(freqdist)
   --[[
      REQUIRES:
         freqdist -> an instance nlp.FreqDist
      EFFECTS:
         Creates an instance of the nlp.MLEProbDist class
         for use in modeling a probability distribution
         which assumes the probability of generating a sample
         is equivalent to the computed frequency of that sample
         in the underlying frequency distribution
   ]]

   ProbDist.__init(self)
   self.freqdist = freqdist
   self.samples = freqdist.samples
   self.N = #freqdist:keys()
end

function MLEProbDist:prob(name)
   --[[
      REQUIRES:
         name -> the name of a sample
      EFFECTS:
         Computes the probability of a given sample
   ]]

   return self.freqdist:freq(name)
end

function MLEProbDist:max()
   --[[
      EFFECTS:
         Computes the sample with
         maximum probability
   ]]

   return self.freqdist:max()
end

-- Useful metatable functions
MLEProbDist.__call     = MLEProbDist.prob
MLEProbDist.__tostring = ProbDist.__tostring
MLEProbDist.__len      = ProbDist.__len

-------- LidstoneProbDist

local LidstoneProbDist, _ = torch.class('nlp.LidstoneProbDist', 'nlp.ProbDist')
LidstoneProbDist.SUM_TO_ONE = false

function LidstoneProbDist:__init(freqdist, gamma, bins)
   --[[
      REQUIRES:
         freqdist -> an instance nlp.FreqDist
         gamma -> a number between 0 and 1
         bins -> number of sample values that can be generated
      EFFECTS:
         Creates an instance of the nlp.LidstoneProbDist class
         for use in modeling a MLE probability distribution
         parameterized by two additional members, gamma and bins.
   ]]

   ProbDist.__init(self)
   self.freqdist = freqdist
   self.samples  = freqdist.samples
   self.N        = #freqdist:keys()
   self.gamma    = gamma
   local fdbins  = freqdist:bins()

   if bins == 0 or (bins == nil and freqdist.N == 0) then
      error('A LidstoneProbDist must have at least one bin')
   end
   if bins ~= nil and bins < fdbins then
      error('LidstoneProbDist requires a bin count >= the FreqDist bin count used to create it')
   end
   if bins == nil then
      bins = fdbins
   end
   self.bins = bins

   self.divisor = self.N + bins * gamma
   if self.divisor == 0 then
      self.gamma = 0
      self.divisor = 1
   end
end

function LidstoneProbDist:prob(name)
   --[[
      REQUIRES:
         name -> the name of a sample
      EFFECTS:
         Computes the probability of a given sample
   ]]

   local v = self.freqdist:freq(name)
   return (v + self.gamma) / self.divisor
end

function LidstoneProbDist:discount()
   --[[
      EFFECTS:
         Returns the computed discount ratio
   ]]

   local gb = self.gamma * self.bins
   return gb / (self.N + gb)
end

function LidstoneProbDist:max()
   --[[
      EFFECTS:
         Computes the sample with
         maximum probability
   ]]

   return self.freqdist:max()
end

-- Useful metatable functions
LidstoneProbDist.__call     = LidstoneProbDist.prob
LidstoneProbDist.__tostring = ProbDist.__tostring
LidstoneProbDist.__len      = ProbDist.__len

-------- ELEProbDist

local ELEProbDist, _ = torch.class('nlp.ELEProbDist', 'nlp.LidstoneProbDist')

function ELEProbDist:__init(fd, bins)
   --[[
      REQUIRES:
         fd -> an instance nlp.FreqDist
         bins -> number of sample values that can be generated
      EFFECTS:
         Creates an instance of the nlp.ELEProbDist class,
         a nlp.LidstoneProbDist with gamma = 0.5
   ]]

   LidstoneProbDist.__init(self, fd, 0.5, bins)
end

-- Useful metatable functions
ELEProbDist.__call     = LidstoneProbDist.prob
ELEProbDist.__tostring = ProbDist.__tostring
ELEProbDist.__len      = ProbDist.__len

-------- ConditionalProbDist

local ConditionalProbDist = torch.class('nlp.ConditionalProbDist')

function ConditionalProbDist:__init(cfdist, pdist_factory, ...)
   --[[
      REQUIRES:
         cfdist -> an instance of nlp.ConditionalFreqDist
         pdist_factory -> a factory function that returns
            an instance (or subclass, of course) of nlp.ProbDist
         ... -> additional arguments passed into pdist_factory
      EFFECTS:
         Creates an instance of the nlp.ConditionalProbDist class
         for use in modeling the probability distributions
         constructed by an underlying frequency distribution
         under a certain condition. 
   ]]

   self.pdist_factory = pdist_factory
   self.pdist_args = {...}
   self.samples = {}
   self.N = 0

   for cond, sample in pairs(cfdist.samples) do
      self.samples[cond] = pdist_factory(cfdist(cond), ...)
      self.N = self.N + 1
   end
end

function ConditionalProbDist:get(cond)
   --[[
      REQUIRES:
         cond -> the name of a condition
      EFFECTS:
         Returns the probability distributed associated
         with a specific condition
   ]]

   if type(cond) == 'table' then
      cond = torch.serialize(cond)
   end
   return self.samples[cond]
end

function ConditionalProbDist:__tostring()
   --[[
      EFFECTS:
         Returns a string representation 
         of a nlp.ConditionalProbDist
   ]]

   return 'nlp.ConditionalProbDist with ' .. self.N .. ' conditions'
end

-- Useful metatable functions
ConditionalProbDist.__call = ConditionalProbDist.get