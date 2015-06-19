local Module     = torch.getmetatable('nn.Module')
local Criterion  = torch.getmetatable('nn.Criterion')
local Sequential = torch.getmetatable('nn.Sequential')
local Container  = torch.getmetatable('nn.Container')

-------- Module

Module.clones = {}
Module.step = 1
Module.clone = rnn.clone

Module.forward = function(self, input)
   --[[
      REQUIRES:
         self -> an instance of nn.Module
         input -> a torch Tensor
      EFFECTS:
         Feeds input through either the network
         or it's clone at the correct time-step
   ]]

   local clone = self.clones[self.step] or self
   return clone:updateOutput(input)
end

Module.backward = function(self, input, gradOutput, scale)
   --[[
      REQUIRES:
         self -> an instance of nn.Module
         input -> a torch Tensor
         gradOutput -> a torch Tensor, output of a criterion
         scale -> a number or nil
      EFFECTS:
         Backpropogates input and gradOutput through 
         either the network or it's clone at the 
         correct time-step
   ]]

   scale = scale or 1
   local clone = self.clones[self.step] or self
   clone:updateGradInput(input, gradOutput)
   clone:accGradParameters(input, gradOutput, scale)
   self.step = self.step + 1
   if self.step > #self.clones then
      self.step = 1
   end
   return clone.gradInput
end

-------- Criterion

Criterion.clones = {}
Criterion.step = 1
Criterion.clone = rnn.clone

Criterion.forward = function(self, input, target)
   --[[
      REQUIRES:
         self -> an instance of nn.Criterion
         input -> a torch Tensor
      EFFECTS:
         Feeds input through either the criterion
         or it's clone at the correct time-step
   ]]

   local clone = self.clones[self.step] or self
   return clone:updateOutput(input, target)
end

Criterion.backward = function(self, input, target)
   --[[
      REQUIRES:
         self -> an instance of nn.Module
         input -> a torch Tensor
         target -> a torch Tensor
      EFFECTS:
         Backpropogates input and target through 
         either the criterion or it's clone at the 
         correct time-step
   ]]

   local clone = self.clones[self.step] or self
   self.step = self.step + 1
   if self.step > #self.clones then
      self.step = 1
   end
   return clone:updateGradInput(input, target)
end

-------- Sequential

Sequential.backward = function(self, input, gradOutput, scale)
   --[[
      REQUIRES:
         self -> an instance of nn.Sequential
         input -> a torch Tensor
         gradOutput -> a torch Tensor, output of a criterion
         scale -> a number or nil
      EFFECTS:
         Backpropogates input and gradOutput through 
         either the sequential or it's clone at the 
         correct time-step
   ]]

   scale = scale or 1
   local currentGradOutput = gradOutput
   local clone = self.clones[self.step] or self
   local nmods = #clone.modules
   local currentModule = clone.modules[nmods]
   for i = nmods - 1, 1, -1 do
      local previousModule = clone.modules[i]
      currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
      currentModule.gradInput = currentGradOutput
      currentModule = previousModule
   end
   currentGradOutput = currentModule:backward(input, currentGradOutput, scale)
   self.gradInput = currentGradOutput
   self.step = self.step + 1
   if self.step > #self.clones then
      self.step = 1
   end

   for i = 1, nmods do
      local mod = clone.modules[i]
      mod.step = mod.step + 1
      if mod.step > #self.clones then
         mod.step = 1
      end
   end
   return currentGradOutput
end

-------- Container

local ContainerParameters = Container.parameters

Container.parameters = function(self)
   --[[
      REQUIRES:
         self -> an instance (or subclass, of course) of nn.Container
      EFFECTS:
         Collects the parameters of a container
         and all it's clone (if it has any)
   ]]

   local cl = #self.clones
   if cl == 0 then
      return ContainerParameters(self)
   else
      local params = {}
      local grads = {}
      for i = 1, cl do
         local ps, gs = ContainerParameters(self.clones[i])
         for n, elem in pairs(ps) do
            table.insert(params, elem)
         end
         for n, elem in pairs(gs) do
            table.insert(grads, elem)
         end
      end
      return params, grads
   end
end