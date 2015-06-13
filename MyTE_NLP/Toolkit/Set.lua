local Set = torch.class('nlp.Set')

function Set:__init(t)
   --[[
      REQUIRES:
         t -> a lua table or nil
      EFFECTS:
         Creates an instance of the nlp.Set class.
         Inside, the nlp.Set class stores its members
         as keys in a lua table with all values
         set to true.
   ]]

   self.values = {}
   if t ~= nil and type(t) == 'table' then
      for i = 1, #t do
         self.values[t[i]] = true
      end
   end
end

function Set:add(val)
   --[[
      REQUIRES:
         val -> a typical key of a lua table,
         ie. a string or number
      EFFECTS:
         If val is already a member of this
         nlp.Set, return false, else add val
         by setting self.values[val] to true
         and returning true.
   ]]

   local v = self.values[val]
   if v then return false end
   self.values[val] = true
   return true
end

function Set:contains(val)
   --[[
      REQUIRES:
         val -> a typical key of a lua table,
         ie. a string or number
      EFFECTS:
         Returns true if val is a member of
         this nlp.Set
   ]]

   local v = self.values[val]
   return v or false
end

function Set:table()
   --[[
      EFFECTS:
         Returns the members of this nlp.Set
         as a lua table with values equal
         to the members
   ]]

   local rv = {}
   for k, v in pairs(self.values) do
      table.insert(rv, k)
   end
   return rv
end

function Set:iter()
   --[[
      EFFECTS:
         Returns a function which returns
         the elements of nlp.Set member-by-member.
         Useful for looping over the members of
         a nlp.Set
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

function Set:__len()
   --[[
      EFFECTS:
         Returns the number of members in a
         nlp.Set
   ]]

   local n = 0
   for k, v in pairs(self.values) do n = n + 1 end
   return n
end

function Set:__tostring()
   --[[
      EFFECTS:
         Returns a string representation of a
         nlp.Set
   ]]

   return 'Set storing ' .. #self .. ' values'
end

-- Useful metatable operation
Set.__call = Set.add