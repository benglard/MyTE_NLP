local Module, parent = torch.class('rnn.Module', 'nn.Module')

function Module:__init(inputSize, hiddenSize, batchSize, seqLength)
   --[[
      REQUIRES:
         inputSize -> a number
         hiddenSize -> a number
         batchSize -> a number or nil
         seqLength -> a number or nil
      EFFECTS:
         Creates an instance of rnn.Module for use
         in architecting recurrent neural network
         models.
   ]]

   parent.__init(self)
   assert(type(inputSize) == 'number' and type(hiddenSize) == 'number',
      'rnn.Module.__init requires input of type number')
   self.inputSize = inputSize
   self.hiddenSize = hiddenSize
   self.batchSize = batchSize or 1
   self.seqSize = seqLength or 1
   self.input = {}
end

function Module:parameters()
   --[[
      EFFECTS:
         Returns the parameters of self.layer
         or self
   ]]

   local mod = self.layer or self
   return mod:parameters()
end

function Module:apply(name, debug)
   --[[
      REQUIRES:
         name -> a lua string
         debug -> a boolean
      EFFECTS:
         Names the recurrent layer and
         sets the debugging mode
   ]]

   self:name(name or '')
   if debug ~= nil then
      self:debug(debug)
   end
   return self
end

function Module:name(s)
   --[[
      REQUIRES:
         s -> a lua string
      EFFECTS:
         Names the recurrent layer and all its
         potential clones      
   ]]

   local cl = #self.clones
   if cl > 0 then
      for i = 1, cl do
         local mod = self.clones[i]
         mod.name = s
      end
   else
      local mod = self.layer or self
      mod.name = s
   end
   return self
end

function Module:annotate()
   --[[
      EFFECTS:
         Calls nngraph.annotateNodes
   ]]

   local ok, err = pcall(nngraph.annotateNodes)
   return ok
end

function Module:debug(val)
   --[[
      EFFECTS:
         Sets the nngraph debugging mode
   ]]

   nngraph.setDebug(val)
   return self
end

function Module:reset(stdv)
   --[[
      REQUIRES:
         stdv -> a number
      EFFECTS:
         Resets a module or one of its clones
   ]]

   local mod = self.clones[self.step] or self.layer or self
   mod:reset(stdv)
end

function Module:updateOutput(input)
   --[[
      REQUIRES:
         input -> a torch Tensor
      EFFECTS:
         Feeds input through either the network
         or it's clone at the correct time-step
   ]]

   local mod = self.clones[self.step] or self.layer or self
   return mod:updateOutput(input)
end

function Module:updateGradInput(input, gradOutput)
   --[[
      REQUIRES:
         input -> a torch Tensor
         gradOutput -> a torch Tensor
      EFFECTS:
         Calculates the gradient with respect to the input
   ]]

   local mod = self.clones[self.step] or self.layer or self
   return mod:updateGradInput(input, gradOutput)
end

function Module:accGradParameters(input, gradOutput, scale)
   --[[
      REQUIRES:
         input -> a torch Tensor
         gradOutput -> a torch Tensor
         scale -> a number
      EFFECTS:
         Calculates the gradient with respect to the
         modules' parameters
   ]]

   local mod = self.clones[self.step] or self.layer or self
   return mod:accGradParameters(input, gradOutput, scale)
end

function Module:clone(T)
   --[[
      REQUIRES:
         T -> a number, representing number of
            timesteps over which to clone the model
      EFFECTS:
         Clones self T times
   ]]

   local mod = self.layer or self
   self.clones = parent.clone(mod, T).clones
   return self
end