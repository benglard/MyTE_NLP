local RNNTrainer = torch.class('rnn.RNNTrainer')

function RNNTrainer:__init(model, criterion, method, data, ...)
   --[[
      REQUIRES:
         model -> an instance of nn.Module
         criterion -> an instance of nn.Criterion
         method -> an optimization method from torch/optim,
            e.g. 'sgd' or 'rmsprop'
         data -> a torch Tensor or lua table
         ... -> extra params to pass into optim[method]
      EFFECTS:
         Creates an instance of rnn.RNNTrainer for use
         in training recurrent neural network models
   ]]

   self.model = model
   self.params, self.grads = model:getParameters()
   self.o = optim[method]
   self.data = data
   self.config = {}
   for key, val in pairs({...}) do
      self.config[key] = val
   end
end

function RNNTrainer:train(params)
   --[[
      REQUIRES:
         params -> a lua table or nil
      EFFECTS:
         Trains a recurrent neural network
         using some specified optimization method
         on some specified training data
   ]]

   params = params or {}
   local maxiter  = params.maxiter or 1
   local progress = params.progress or false

   local save  = params.save or false
   local every = params.every or 1
   local name  = params.filename or 'model.save'

   local isTensor = self.data.isTensor ~= nil
   local size = 0
   if isTensor then
      size = self.data:size(1)
   else
      size = #self.data
   end
   local epoch = 1
   local shuffledIndices = torch.randperm(size, 'torch.LongTensor')

   while epoch < maxiter + 1 do
      for t = 1, size do
         if progress then 
            xlua.progress(t, size)
         end

         local idx = shuffledIndices[t]
         local input = self.data[idx][1] or self.data[idx].x
         local label = self.data[idx][2] or self.data[idx].y

         local function _train(x)
            if x ~= self.params then self.params:copy(x) end
            self.grads:zero()
            local output = self.model:forward(input)
            local err = self.criterion:forward(output, label)
            local gradOutput = self.criterion:backward(output, label)
            self.model:backward(input, gradOutput)
            return err, self.grads
         end

         self.o(_train, self.params, self.config)
      end

      if save and (epoch % every == 0) then
         torch.save(name, self.model)
      end

      epoch = epoch + 1
   end
end