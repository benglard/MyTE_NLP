local DQEncDec, parent = torch.class('rl.DQEncDec', 'rnn.EncDec')
local RNNModule = torch.getmetatable('rnn.Module')

function DQEncDec:__init(encoder, decoder, stop, input, hidden, batch, seq, vocab)
   --[[
      REQUIRES:
         encoder -> an instance of nn.Module or nngraph.gModule
         decoder -> an instance of nn.Module or nngraph.gModule
         stop -> stop symbol
         input -> a number
         hidden -> a number
         batch -> a number or nil
         seq -> a number or nil
         vocab -> size of vocabulary
      EFFECTS:
         Creates an instance of the rnn.EncDec class for use
         in building recurrent nueral network
         encoder-decoder models.
   ]]

   RNNModule.__init(self, input, hidden, batch, seq)
   self.encoder = encoder:clone(self.seqSize)
   self.decoder = decoder

   self.layer = nn.Sequential()
   self.layer:add(encoder)
   self.layer:add(decoder)

   self.stop = stop
   self.vocabSize = vocab
   self:restart()

   self.encw, self.encdw = self.encoder:getParameters()
end

function DQEncDec:updateOutput(input)
   --[[
      REQUIRES:
         input -> a torch Tensor
      EFFECTS:
         Feeds input through either the encoder
         or decoder at the correct time-step
   ]]

   local es = self.step.encoder
   local ds = self.step.decoder
   local enc = self.encoder.clones[es]
   local stop = input == self.stop

   if stop and (not self.estop or es == self.seqSize) then
      input = torch.Tensor(self.batchSize):typeAs(self.prev):fill(self.vocabSize)
      self.inputs[es]:copy(input)
      self.estop = true
      return enc:forward(input)
   elseif stop and (self.estop or ds == self.seqSize) then
      self.estop = false
      return nil
   elseif es < self.seqSize and (not self.estop) then
      self.inputs = self.inputs:typeAs(input)
      self.inputs[es]:copy(input)
      local output = enc:forward(input)
      self.step.encoder = es + 1
      self.prev:typeAs(output):resizeAs(output):copy(output)
      return output
   elseif ds < self.seqSize then
      return self.decoder:forward(input)
   end
end

function DQEncDec:backward(reward, config)
   --[[
      REQUIRES:
         reward -> a number representing the agent's
         reward after performing an action
      EFFECTS:
         config -> a table of options to pass into
            optim.sgd
   ]]

   config = config or {}

   self.decoder:backward(reward)
   local currentGradOutput = self.decoder.gradInput:clone()

   local ds = self.step.decoder
   if ds == 1 then
      local loss = self.decoder.loss
      self.encdw:zero()

      local es = self.step.encoder
      for i = es, 1, -1 do
         local encinput = self.inputs[i]
         local enc = self.encoder.clones[i]
         enc:backward(encinput, currentGradOutput)
      end

      local feval = function(x) 
         if x ~= self.encw then self.encw:copy(x) end
         return loss, self.encdw
      end
      optim.sgd(feval, self.encw, config)
   end

   self.gradInput = currentGradOutput
   self.step.decoder = ds + 1
end

function DQEncDec:restart()
   --[[
      EFFECTS:
         Reloads the model to initial
         values
   ]]

   self.estop = false
   self.step = {
      encoder = 1,
      decoder = 1
   }

   self.prev = torch.Tensor()
   self.inputs = torch.zeros(self.seqSize, self.batchSize, self.inputSize)
end

function DQEncDec:__tostring__()
   --[[
      EFFECTS:
         Returns the string representation of
         self.layer
   ]]

   return 'rl.DQEncDec: ' .. tostring(self.layer)
end