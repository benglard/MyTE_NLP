local EncDec, parent = torch.class('rnn.EncDec', 'rnn.Module')

function EncDec:__init(encoder, decoder, stop, input, hidden, batch, seq)
   --[[
      REQUIRES:
         encoder -> an instance of nn.Module or nngraph.gModule
         decoder -> an instance of nn.Module or nngraph.gModule
         input -> a number
         hidden -> a number
         batch -> a number or nil
         seq -> a number or nil
         stop -> stop symbol
      EFFECTS:
         Creates an instance of the rnn.EncDec class for use
         in building recurrent nueral network
         encoder-decoder models.
   ]]

   parent.__init(self, input, hidden, batch, seq)
   self.encoder = encoder:clone(self.seqSize)
   self.decoder = decoder:clone(self.seqSize)

   self.layer = nn.Sequential()
   self.layer:add(encoder)
   self.layer:add(decoder)

   self.stop = stop
   self:restart()
end

function EncDec:updateOutput(input)
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
   local dec = self.decoder.clones[ds]

   local stop = input == self.stop
   if stop and (not self.estop or es == self.seqSize) then
      input = torch.Tensor(self.batchSize, self.inputSize):typeAs(self.prev)
      for i = 1, self.batchSize do
         input[i][self.inputSize] = 1.0
      end
      self.inputs[es]:copy(input)
      dec.modules[1].prev_h[1]:copy(self.prev)
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
      self.prev:resizeAs(output):typeAs(output):copy(output)
      return output
   elseif ds < self.seqSize then
      return dec:forward(input)
   end
end

function EncDec:backward(input, gradOutput, scale)
   --[[
      REQUIRES:
         input -> a torch Tensor
         gradOutput -> a torch Tensor, output of a criterion
         scale -> a number or nil
      EFFECTS:
         Backpropogates input and gradOutput through 
         the decoder and encoder at the correct time step
   ]]

   scale = scale or 1
   local currentGradOutput = gradOutput
   local ds = self.step.decoder
   local dec = self.decoder.clones[ds]

   currentGradOutput = dec:backward(input, currentGradOutput, scale)
   dec.gradInput = currentGradOutput

   local es = self.step.encoder - (ds - 1)
   if es > 0 then
      local encinput = self.inputs[es]
      local enc = self.encoder.clones[es]
      currentGradOutput = enc:backward(encinput, currentGradOutput, scale)
   end

   self.gradInput = currentGradOutput
   self.step.decoder = ds + 1
end

function EncDec:decode(input)
   --[[
      REQUIRES:
         input -> a torch Tensor
      EFFECTS:
         Feeds input through the decoder
         or it's clone at the correct time-step.
         But returns the output of only
         the recurrent layer (so it can be fed back)
   ]]

   self:updateOutput(input)
   return self:state()
end

function EncDec:state()
   --[[
      EFFECTS:
         Returns hidden state of first layer
         of decoder
   ]]

   local ds = self.step.decoder - 1
   if ds == -1 then ds = 1 end
   return self.decoder.clones[ds].modules[1].output:clone()
end

function EncDec:restart()
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

function EncDec:__tostring__()
   --[[
      EFFECTS:
         Returns the string representation of
         self.layer
   ]]

   return tostring(self.layer)
end

EncDec.encode = EncDec.updateOutput