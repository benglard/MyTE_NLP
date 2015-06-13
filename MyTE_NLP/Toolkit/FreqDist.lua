-------- DefaultTable

local DefaultTable = torch.class('nlp.DefaultTable')

function DefaultTable:__init(default)
   --[[
      REQUIRES:
         default -> a piece of data or function
      EFFECTS:
         Creates an instance of the nlp.DefaultTable class,
         a wrapper around the usual lua table so that
         keys with nil values are replaced by a given
         default value.
   ]]

   self.default = default
   if type(default) == 'table' then
      local mt = getmetatable(default)
      if mt.__init ~= nil then
         self.isfunc = true
      end
      self.isfunc = true
   elseif type(default) == 'function' then
      self.isfunc = true
   end
   self.samples = {}
end

function DefaultTable:get(key)
   --[[
      REQUIRES:
         key -> a typical lua string or table,
         or a table to be serialized
      EFFECTS:
         Returns the value of that key or
         the default value
   ]]

   if type(key) == 'table' then
      key = torch.serialize(key)
   end

   local v = self.samples[key]
   if v == nil then
      if self.isfunc then
         local def = self.default()
         self.samples[key] = def
         return def
      else
         self.samples[key] = self.default
         return self.default
      end
   else
      return v
   end
end

function DefaultTable:push(key, value)
   --[[
      REQUIRES:
         key -> a typical lua key
         value -> a typical lua value
      EFFECTS:
         Sets the key to match the value
   ]]

   self.samples[key] = value
end

function DefaultTable:__call(key, value)
   --[[
      REQUIRES:
         key -> a typical lua key
         value -> a typical lua value or nil
      EFFECTS:
         If value is nil, returns the value at
         the given key, else push the key-value
         pair.
   ]]

   if value == nil then
      return self:get(key)
   else
      self:push(key, value)
   end
end

function DefaultTable:__len()
   --[[
      EFFECTS:
         Returns the length of a nlp.DefaultTable, given as
         the number of keys.
   ]]

   local n = 0
   for k, v in pairs(self.samples) do n = n + 1 end
   return n
end

function DefaultTable:__tostring()
   --[[
      EFFECTS:
         Returns a string representation of a
         nlp.DefaultTable
   ]]

   return 'nlp.DefaultTable with ' .. #self .. ' samples'
end

-------- Counter

local Counter = torch.class('nlp.Counter')

function Counter:__init(input)
   --[[
      REQUIRES:
         input -> a lua table or nlp.String
      EFFECTS:
         Creates an instance of the nlp.Counter class,
         useful for counting a collection of objects.
   ]]

   self.samples = {}
   self.N = 0

   if input == nil then
      return self
   end

   if torch.type(input) == 'table' then
      local array = input[#input] ~= nil
      if array then
         for n, elem in pairs(input) do
            local current = self.samples[elem] or 0
            current = current + 1
            self.samples[elem] = current
            self.N = self.N + 1
         end
      else
         for k, v in pairs(input) do
            self.samples[k] = v
            self.N = self.N + v
         end
      end
   elseif torch.type(input) == 'nlp.String' then
      for char in input:iter() do
         local current = self.samples[char] or 0
         current = current + 1
         self.samples[char] = current
         self.N = self.N + 1
      end
   else
      error('FreqDist.__init requires input of type table or nlp.String')
   end
end

function Counter:__len()
   --[[
      EFFECTS:
         Returns the length of a nlp.Counter, given as
         the number of keys.
   ]]

   local n = 0
   for k, v in pairs(self.samples) do n = n + 1 end
   return n
end

function Counter:keys()
   --[[
      EFFECTS:
         Returns the keys
   ]]

   local rv = {}
   for k, v in pairs(self.samples) do
      table.insert(rv, k)
   end
   return rv
end

function Counter:values()
   --[[
      EFFECTS:
         Returns the values
   ]]

   local rv = {}
   for k, v in pairs(self.samples) do
      table.insert(rv, v)
   end
   return rv
end

function Counter:get(key, default)
   --[[
      EFFECTS:
         Returns the value at a given key
         or a default value
   ]]

   return self.samples[key] or default
end

function Counter:push(key, value)
   --[[
      EFFECTS:
         Increments the value at a given key
   ]]

   local current = self:get(key, 0)
   current = current + 1
   self.samples[key] = value or current
   self.N = self.N + 1
end

function Counter:__tostring()
   --[[
      EFFECTS:
         Returns a string representation of
         a nlp.Counter
   ]]

   return torch.type(self) .. ' with ' .. #self .. ' keys storing ' .. self.N .. ' values'
end

-- Useful metatable functions
Counter.__call  = Counter.get

-------- FreqDist

local FreqDist, _ = torch.class('nlp.FreqDist', 'nlp.Counter')

function FreqDist:get(name)
   --[[
      EFFECTS:
         Returns the value at a given key or 0
   ]]

   return self.samples[name] or 0
end

function FreqDist:freq(name)
   --[[
      EFFECTS:
         Returns the computed frequency
         of a given sample
   ]]

   if self.N == 0 then return 0 end
   return self:get(name) / self.N
end

function FreqDist:bins()
   --[[
      EFFECTS:
         Returns the number of non-zero samples
   ]]

   local n = 0
   for name, val in pairs(self.samples) do
      if val > 0 then n = n + 1 end
   end
   return n
end

function FreqDist:max()
   --[[
      EFFECTS:
         Returns the sample with maximum value
   ]]

   local name, max = next(self.samples)
   for key, val in pairs(self.samples) do
      if val > max then
         max = val
         name = key
      end
   end
   return name
end

-- Useful metatable functions
FreqDist.__call     = FreqDist.get
FreqDist.__len      = Counter.__len
FreqDist.__tostring = Counter.__tostring

-------- ConditionalFreqDist

local ConditionalFreqDist, _ = torch.class('nlp.ConditionalFreqDist', 'nlp.DefaultTable')

function ConditionalFreqDist:__init(samples)
   --[[
      REQUIRES:
         samples -> a lua table or nil
      EFFECTS:
         Creates an instance of nlp.ConditionalFreqDist for
         use in modeling a collection of frequency distributions
         constructed under different conditions
   ]]

   DefaultTable.__init(self, nlp.FreqDist)
   samples = samples or {}
   for cond, sample in pairs(samples) do
      self:push(cond, sample)
   end
end

function ConditionalFreqDist:push(cond, sample)
   --[[
      REQUIRES:
         cond -> a condition, typical key or nlp.DefaultTable
         sample -> a sample result
      EFFECTS:
         Gets (or creates) the FreqDist for a given condition
         and pushes the sample result onto it
   ]]

   self:get(cond):push(sample)
end

function ConditionalFreqDist:__tostring()
   --[[
      EFFECTS:
         Returns a string representation of
         a nlp.ConditionalFreqDist
   ]]

   return 'nlp.ConditionalFreqDist with ' .. #self .. ' conditions'
end

-- Useful metatable functions
ConditionalFreqDist.__call = ConditionalFreqDist.get
ConditionalFreqDist.__len  = DefaultTable.__len