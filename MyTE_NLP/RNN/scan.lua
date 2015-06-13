local elemApply = function(elem, cb1, cb2, ...)
   --[[
      REQUIRES:
         elem -> a torch Tensor or lua table
         cb1 -> a function that takes elem and ...
            as input
         cb2 -> a function that takes elem and ...
            as input
         ... -> extra params to cb1 or cb2
      EFFECTS:
         If elem is a torch Tensor, calls cb1(elem, ...),
         else calls cb2(elem, ...)
   ]]

   local isTensor = elem.isTensor ~= nil
   if isTensor then
      return cb1(elem, ...)
   elseif type(elem) == 'table' then
      return cb2(elem, ...)
   else
      error('sequence must be of type torch.Tensor or lua table')
   end
end

return function(options)
   --[[
      REQUIRES:
         options -> a lua table or nil
      EFFECTS:
         Applies a function over a set of
         sequences (torch Tensors or lua tables)
   ]]

   options = options or {}
   local fn = options.fn or function(...) end
   local seqs = options.sequences or {}
   local outputs = options.outputs or {}
   local nonseqs = options.nonsequences or {}
   local nsteps = options.nsteps or nil

   local tensorSize = function(e) return e:size(1) end
   local tableSize = function(e) return #e end

   if nsteps == nil then
      nsteps = elemApply(seqs[1], tensorSize, tableSize)
      for i = 2, #seqs do
         local size = elemApply(seqs[i], tensorSize, tableSize)
         if nsteps > size then
            nsteps = size
         end
      end
   end

   local identity = function(e) return e end
   local getElem = function(e, ...) return e[...] end
   local function getinputs(i, src, dest)
      for j = 1, #src do
         local elem = src[j]
         table.insert(dest, elemApply(elem, identity, getElem, i))
      end
   end

   local rv = {}
   for i = 1, nsteps do
      local inputs = {i}
      getinputs(i, seqs, inputs)
      getinputs(i, outputs, inputs)
      getinputs(i, nonseqs, inputs)

      local res = {fn(unpack(inputs))}
      table.insert(rv, res)
      outputs = res
   end
   return rv
end