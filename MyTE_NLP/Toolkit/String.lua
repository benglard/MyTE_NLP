local String = torch.class('nlp.String')

function String:__init(str)
   --[[
      REQUIRES:
         str -> a string or nlp.String
      EFFECTS:
         Creates an instance of the nlp.String class
         for use in nlp string operations.
   ]]

   if torch.type(str) == 'string' then
      self.str = str
   elseif torch.type(str) == 'nlp.String' then
      self.str = str.str
   else
      error('String.__init requires argument of type string or nlp.String')
   end
end

function String:__tostring()
   --[[
      EFFECTS:
         Returns the lua string associated
         with this nlp.String object
   ]]

   return self.str
end

function String:__len()
   --[[
      EFFECTS:
         Returns the length of the lua string
         associated with this nlp.String object
   ]]

   return #self.str
end

function String:get(key)
   --[[
      REQUIRES:
         key -> a number or lua table
      EFFECTS:
         Indexes an instance of nlp.String by key.
         
         If key is a number, String.get returns the
         char at the key'th position in the string.
         
         If key is a table with 3 elements, String.get
         returns the char at every 3rd element of the table
         between the first and second elements of the table.
         
         If key is a table with 2 elements, String.get
         returns every char between those two elements.

         If key is a table with 1 element, String.get
         returns the char at the key'th position in the string.
   ]]

   if type(key) == 'number' then
      return nlp.String(self.str:sub(key, key))
   elseif type(key) == 'table' then
      local idx1, idx2, skip = unpack(key)
      if idx2 == nil then
         return nlp.String(self.str:sub(idx1, idx1))
      end
      if skip == nil then
         return nlp.String(self.str:sub(idx1, idx2))
      else
         local s = ''
         for i = idx1, idx2, skip do
            s = s .. self.str:sub(i, i)
         end
         return nlp.String(s)
      end
   end
end

function String:reverse()
   --[[
      EFFECTS:
         Reverses an nlp.String
   ]]

   local len = #self
   return self{len, 1, -1}
end

function String:run(method, ...)
   --[[
      REQUIRES:
         method -> a function from the builtin
         lua string library, e.g. string.upper
         ... -> the usual parameters of the
         lua string method
      EFFECTS:
         Applies method to the lua string associated
         with this object. If the return value is a
         lua string, return a new nlp.String with
         that value, else return that value.
   ]]

   local rv = method(self.str, ...)
   if type(rv) == 'string' then
      return nlp.String(rv)
   else
      return rv
   end
end

function String:add(other)
   --[[
      REQUIRES:
         other -> a lua string or nlp.String
      EFFECTS:
         Returns the concatenation of self
         and other as an nlp.String
   ]]

   local t = torch.type(other)
   if t == 'string' then
      return nlp.String(self.str .. other)
   elseif t == 'nlp.String' then
      return nlp.String(self.str .. other.str)
   else
      error('String.__add requires argument of type string or nlp.String')
   end
end

function String:mul(other)
   --[[
      REQUIRES:
         other -> a number
      EFFECTS:
         Repeats an nlp.String other times, ie.
         nlp.String('yo') * 3 = nlp.String('yoyoyo')
   ]]

   assert(type(other) == 'number',
      'String.__mul requires argument of type number')
   local rv = self
   for i = 1, other - 1 do
      rv = rv + self
   end
   return rv
end

function String:table()
   --[[
      EFFECTS:
         Returns the nlp.String as a table
         with the values equivalent to the
         chars of the associated lua string.
   ]]

   local rv = {}
   for i = 1, #self do
      rv[i] = self(i).str
   end
   return rv
end

function String:iter()
   --[[
      EFFECTS:
         Returns a function which returns
         the elements of nlp.String char-by-char.
         Useful for looping over the chars of
         a nlp.String
   ]]

   local t = self:table()
   local n = 0
   local size = #t
   return function()
      n = n + 1
      if (t and n <= size) then return t[n]
      else return nil end
   end
end

-- Useful metatable functions
String.__call = String.get
String.__add  = String.add
String.__mul  = String.mul

--[[ 
Allow every function from the string library
(e.g. upper, gsub, match) to be called on an
nlp.String object using the same typical
parameters, by actually running nlp.String.run
with that function and its arguments.
]]

for name, func in pairs(string) do
   String[name] = function(self, ...)
      return self:run(func, ...)
   end
end