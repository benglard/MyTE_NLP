return function(self, T, verbose)
   --[[
      REQUIRES:
         self -> an instance of nn.Module or nngraph.gModule
            or nn.Criterion
         T -> a number, representing number of
            timesteps over which to clone the model
      EFFECTS:
         Clones self T times
   ]]

   verbose = verbose or false

   self.clones = {}
   local params, gradParams
   if self.parameters then
      params, gradParams = self:parameters()
      if params == nil then
         params = {}
      end
   end

   local paramsNoGrad
   if self.parametersNoGrad then
      paramsNoGrad = self:parametersNoGrad()
   end

   local mem = torch.MemoryFile('w'):binary()
   mem:writeObject(self)

   for t = 1, T do
      if verbose then print('clone #'.. t) end
      -- We need to use a new reader for each clone.
      -- We don't want to use the pointers to already read objects.
      local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
      local clone = reader:readObject()
      reader:close()

      if self.parameters then
         local cloneParams, cloneGradParams = clone:parameters()
         local cloneParamsNoGrad
         for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
         end
         if paramsNoGrad then
            cloneParamsNoGrad = clone:parametersNoGrad()
            for i =1, #paramsNoGrad do
               cloneParamsNoGrad[i]:set(paramsNoGrad[i])
            end
         end
      end
      
      self.clones[t] = clone
      collectgarbage()
   end
   mem:close()
   return self
end