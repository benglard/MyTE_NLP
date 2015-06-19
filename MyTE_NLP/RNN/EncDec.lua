local EncDec, parent = torch.class('rnn.EncDec', 'rnn.Module')

function EncDec:__init(encoder, decoder, inputSize, seqSize, stop)
   --[[
      REQUIRES:
         encoder -> an instance of nn.Module or nngraph.gModule
         decoder -> an instance of nn.Module or nngraph.gModule
         inputSize -> size of input sequence
         seqSize -> size of input sequence
         stop -> stop symbol
      EFFECTS:
         Creates an instance of the rnn.EncDec class for use
         in building recurrent nueral network
         encoder-decoder models.
   ]]

   self.encoder = encoder:clone(seqSize)
   self.decoder = decoder:clone(seqSize)

   self.layer = nn.Sequential()
   self.layer:add(encoder)
   self.layer:add(decoder)

   self.seqSize = seqSize
   self.stop = stop
   self.estop = false

   self.step = {
      encoder = 1,
      decoder = 1
   }

   self.prev = torch.Tensor()
   self.inputs = torch.Tensor(seqSize, inputSize)
end

function EncDec:updateOutput(input)
   --[[
      REQUIRES:
         input -> a torch Tensor
      EFFECTS:
         Feeds input through either the network
         or it's clone at the correct time-step
   ]]

   local es = self.step.encoder
   local ds = self.step.decoder
   local enc = self.encoder.clones[es]
   local dec = self.decoder.clones[ds]

   local stop = input == self.stop
   if stop and (not self.estop or es == self.seqSize) then
      print 'enc stop'
      dec.modules[1].prev_h[1]:copy(self.prev)
      self.estop = true
      return output
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
      local output = dec:forward(input)
      return output
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
         either the sequential or it's clone at the 
         correct time-step
   ]]

   scale = scale or 1
   local currentGradOutput = gradOutput
   local ds = self.step.decoder
   local encinput = self.inputs[ds]
   local enc = self.encoder.clones[ds]
   local dec = self.decoder.clones[ds]

   currentGradOutput = dec:backward(input, currentGradOutput, scale)
   dec.gradInput = currentGradOutput
   self.gradInput = enc:backward(encinput, currentGradOutput, scale)
   self.step.decoder = ds + 1
end

function EncDec:__tostring__()
   --[[
      EFFECTS:
         Returns the string representation of
         self.layer
   ]]

   return tostring(self.layer)
end