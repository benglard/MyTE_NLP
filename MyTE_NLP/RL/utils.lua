torch.Tensor.adddim = function(self, axis)
   --[[
      REQUIRES:
         self -> a torch Tensor
         axis -> a number
      EFFECTS:
         Adds a dimension to self at the axis
         specified by axis
   ]]

   local d = self:dim()
   axis = axis or d + 1

   local sizes = {}
   for i = 1, d do
      if axis == d then
         table.insert(sizes, 1)
      end
      local s = self:size(i)
      table.insert(sizes, s)
   end
   if axis == d + 1 then
      table.insert(sizes, 1)
   end
   self:resize(unpack(sizes))
   return self
end

local function _batchedop(res, t1, t2, op)
   --[[
      REQUIRES:
         res -> a torch Tensor to in which to store the result
         t1 -> a torch Tensor
         t2 -> a torch Tensor
         op -> a torch Tensor operation, e.g. torch.Tensor.add
      EFFECTS: 
         Performs a batched version of the operation
         specified by op on tensors t1 and t2, and stores
         the result in res.
   ]]

   local t1d = t1:dim()
   local t2d = t2:dim()
   local t2l2d = true
   local batch = t1:size(1)
   local range, third

   if t1d == 3 or t2d == 3 then
      if t1d < t2d then
         third = t2:size(1)
         range = t2:size(2)
      else
         range = t1:size(2)
         third = t1:size(3)
      end
      t2l2d = false
   elseif t2d == 1 then
      range = t2:size(1)
   elseif t2d == 2 then
      range = t2:size(2)
      t2 = t2:reshape(range)
   end

   if t2l2d then
      if res then
         res:resize(batch, range)
      else
         res = torch.zeros(batch, range)
      end
      for i = 1, batch do
         op(res[i]:copy(t2), t1[i])
      end
   else
      if t1d < t2d then
         if res then
            res:resize(third, range, batch)
         else
            res = torch.zeros(third, range, batch)
         end
         for i = 1, third do
            for j = 1, range do
               local place = res[i][j]
               place:copy(t1)
               op(place, t2[i][j]:squeeze())
            end
         end
      else
         if res then
            res:resize(batch, range, third)
         else
            res = torch.zeros(batch, range, third)
         end
         for i = 1, batch do
            for j = 1, range do
               local place = res[i][j]
               place:copy(t1[i][j])
               op(place, t2[i])
            end
         end         
      end
   end
   return res
end

local function batchedadd(t1, t2, res)
   -- Performs a batched add
   return _batchedop(res, t1, t2, torch.Tensor.add)
end

local function batchedmul(t1, t2, res)
   -- Performs a batched mul
   return _batchedop(res, t1, t2, torch.Tensor.mul)
end

local function batcheddiv(t1, t2, res)
   -- Performs a batched div
   return _batchedop(res, t1, t2, torch.Tensor.div)
end

-- Utility functions
torch.badd = batchedadd
torch.bmul = batchedmul
torch.bdiv = batcheddiv

torch.Tensor.badd = function(self, t2)
   --[[
      REQUIRES:
         self -> a torch Tensor
         t2 -> a torch Tensor
      EFFECTS:
         Perform a batched add on self
         and t2 and copy the result
         to self
   ]]

   self:copy(batchedadd(self, t2, nil))
   return self
end

torch.Tensor.bmul = function(self, t2)
   --[[
      REQUIRES:
         self -> a torch Tensor
         t2 -> a torch Tensor
      EFFECTS:
         Perform a batched mul on self
         and t2 and copy the result
         to self
   ]]

   self:copy(batcheddiv(self, t2, nil))
   return self
end

torch.Tensor.bdiv = function(self, t2)
   --[[
      REQUIRES:
         self -> a torch Tensor
         t2 -> a torch Tensor
      EFFECTS:
         Perform a batched div on self
         and t2 and copy the result
         to self
   ]]

   self:copy(batcheddiv(self, t2, nil))
   return self
end

local tts = torch.Tensor.sum
local function sum(t, axis)
   -- weird sum to allow negative axes
   axis = axis or 1
   if axis >= 0 then return tts(t, axis) end
   if axis == -1 then
      local size = t:size(2)
      local sums = {}
      for i = size, 1, -1 do
         table.insert(sums, tts(t[{{}, i}]))
      end
      return torch.Tensor(sums):adddim(1)
   end
end

torch.sum = sum
torch.Tensor.sum = sum