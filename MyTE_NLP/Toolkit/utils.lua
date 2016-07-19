local tds = require 'tds'

table.reverse = function(t, inplace)
   --[[
      REQUIRES:
         t -> a lua table to be reversed
         inplace -> if true, applies the reversal
            to the original table, t
      EFFECTS:
         Reverses a lua table
   ]]

   assert(type(t) == 'table', 'table.reverse requires input of type table')
   local rv = {}
   local l = #t
   for i = l, 1, -1 do
      rv[l - i + 1] = t[i] 
   end
   if inplace then
      for k, v in pairs(rv) do
         t[k] = v
      end
   else
      return rv
   end
end

table.push = function(t, elem)
   --[[
      REQUIRES:
         t -> a lua table
         elem -> a piece of data
      EFFECTS:
         Pushes elem to the front (index 1)
         of table, t, by reversing t inplace,
         inserting elem at the end of t,
         and reversing t inplace again.
   ]]

   assert(type(t) == 'table', 'table.push requires input of type table')
   table.reverse(t, true)
   table.insert(t, elem)
   table.reverse(t, true)
end

table.clone = function(t)
   --[[
      REQUIRES:
         t -> a lua table
      EFFECTS:
         Creates a clone of t
   ]]

   assert(type(t) == 'table', 'table.clone requires input of type table')
   local rv = {}
   for key, value in pairs(t) do
      rv[key] = value
   end
   return rv
end

table.deepcopy = function(src, dest)
   --[[
      REQUIRES:
         src -> a lua table
         dest -> a lua table or nil
      EFFECTS:
         Performs a deepcopy from src to dest,
         ie. copies all the elements from
         src to dest.
   ]]

   assert(type(src) == 'table', 'table.deepcopy requires input of type table')
   dest = dest or {}
   for k, v in pairs(src) do
      if type(v) == 'table' then
         dest[k] = {}
         table.deepcopy(dest[k], v)
      else
         dest[k] = v
      end
   end
   return setmetatable(dest, getmetatable(src))
end

table.index = function(t, elem)
   --[[
      REQUIRES:
         t -> a lua table
         elem -> a piece of data
      EFFECTS:
         Finds the key of t, for which
         t[key] == value or returns -1
   ]]

   assert(type(t) == 'table', 'table.index requires input of type table')
   for k, v in pairs(t) do
      if v == elem then
         return k
      end
   end
   return -1
end

table.contains = function(t, elem)
   --[[
      REQUIRES:
         t -> a lua table
         elem -> a piece of data
      EFFECTS:
         Returns true if elem is a value
         for which a key exists in t, else
         returns false
   ]]

   return table.index(t, elem) ~= -1
end

table.testtrainsplit = function(t, n)
   --[[
      REQUIRES:
         t -> a lua table
         n -> index to split at
      EFFECTS:
         Splits a lua table of data into
         two tables, one for training data
         and one for testing data
   ]]

   local rv1 = {}
   local rv2 = {}
   for i = 1, n do
      table.insert(rv1, t[i])
   end
   for i = n + 1, #t do
      table.insert(rv2, t[i])
   end
   return rv1, rv2
end

torch.cossim = function(v1, v2)
   --[[
      REQUIRES:
         v1 -> a torch Tensor
         v2 -> a torch Tensor
      EFFECTS:
         Returns the cosine similarity
         of the two vectors, v1 and v2
         by computing dot(v1, v2) / mag(v1) * mag(v2)
   ]]

   local dot = torch.dot(v1, v2)
   local mag = torch.norm(v1, 2) * torch.norm(v2, 2)
   if mag == 0 then return 0
   else return dot / mag end
end

torch.diagsvd = function(s, d1, d2)
   --[[
      REQUIRES:
         s -> a torch Tensor, likely a column vector
            representing the singular values of a larger
            matrix
         d1 -> the width of the original matrix
         d2 -> the height of the original matrix
      EFFECTS:
         Returns a sort of diagonalized form of the singular
         values of matrix, ie. returns a matrix of the same
         size, with the first nrows * ncols a diagonal matrix
         with the singular values on the diagonal.
   ]]

   local n = s:size(1)
   local S = torch.Tensor():resize(d1, d2):zero()
   S[{{1, n}, {1, n}}]:copy(torch.diag(s))
   return S
end

torch.shuffle = function(data, inplace)
   --[[
      REQUIRES:
         data -> a torch Tensor or lua table
         inplace -> if true, will applies the shuffle
            to the original Tensor or table, data
      EFFECTS:
         Shuffles the elements of data along the
         first dimension if data is a tensor, or
         along the only "dimension" if data is a table. 
   ]]

   if inplace == nil then inplace = true end

   local isTensor = torch.isTensor(data)
   local t = torch.type(data)
   assert(t == 'table' or t == 'tds_hash' or isTensor,
      'torch.shuffle requires input of type table, tds.hash,  or tensor')

   local size, rv
   if isTensor then
      size = data:size(1)
      rv = torch.Tensor():typeAs(data):resizeAs(data)
   elseif t == 'table' then
      size = #data
      rv = {}
   elseif t == 'tds_hash' then
      size = #data
      rv = tds.hash()
   end

   local shuffled = torch.randperm(size, 'torch.LongTensor')
   if isTensor then
      for n = 1, size do
         local s = shuffled[n]
         rv[n]:copy(data[s])
      end
   else
      for n = 1, size do
         local s = shuffled[n]
         rv[n] = data[s]
      end
   end

   if inplace then
      if isTensor then
         for n = 1, size do
            local d = rv[n]
            data[n]:copy(d)
         end
      else
         for n = 1, size do
            data[n] = rv[n]
         end
      end
   end

   return rv
end