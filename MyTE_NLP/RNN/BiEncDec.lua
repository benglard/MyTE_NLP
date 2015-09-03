local BiEncDec, Module = torch.class('rnn.BiEncDec', 'rnn.EncDec')

function BiEncDec:updateOutput(input)
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
      input = torch.Tensor(self.inputSize):typeAs(self.prev):fill(self.vocabSize)
      self.inputs[es]:copy(input)
      self.estop = true
      local erv = enc:forward(input)

      local hs = self.decoder.clones[1].modules[1].hiddenSize / 2
      local h_t = torch.zeros(2 * hs)
      h_t[{{1, hs}}]:copy(self.prev)

      -- Build backward encodings automagically
      for i = es, 2, -1 do
         local input = self.inputs[es]:clone():resize(self.inputSize)
         local output = self.encoder.clones[i]:forward(input)
         self.prev = self.prev:typeAs(output):resizeAs(output):copy(output)
      end
      h_t[{{hs + 1, 2 * hs}}]:copy(self.prev)

      -- copy h_f||h_b hidden state to decoder
      self.decoder.clones[1].modules[1].prev_h:copy(h_t)

      local input_1 = self.inputs[1]:clone():resize(self.inputSize)
      local drv = self.encoder.clones[1]:forward(input_1)

      local rv = torch.zeros(2 * hs)
      rv[{{1, hs}}]:copy(erv)
      rv[{{hs + 1, 2 * hs}}]:copy(drv)
      return rv
   elseif stop and (self.estop or ds == self.dseqSize) then
      self.estop = false
      return nil
   elseif es < self.seqSize and (not self.estop) then
      self.inputs = self.inputs:typeAs(input)
      self.inputs[es]:copy(input)
      local output = enc:forward(input)
      self.step.encoder = es + 1
      self.prev = self.prev:typeAs(output):resizeAs(output):copy(output)
      return output
   elseif ds < self.dseqSize then
      if not self.train then
         self.step.decoder = ds + 1
      end
      return dec:forward(input)
   end
end

function BiEncDec:backward(input, gradOutput, scale)
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
   local ds = self.step.decoder
   local dec = self.decoder.clones[ds]
   local currentGradOutput = dec:backward(input, gradOutput, scale)
   dec.gradInput = currentGradOutput

   if ds == 1 then
      local es = self.step.encoder
      -- Backward on backward encodings
      for i = 1, es do
         local encinput = self.inputs[i]
         local enc = self.encoder.clones[i]
         enc:backward(encinput, currentGradOutput, scale)
      end

      -- Backward on forward encodings
      for i = es, 1, -1 do
         local encinput = self.inputs[i]
         local enc = self.encoder.clones[i]
         enc:backward(encinput, currentGradOutput, scale)
      end
   end

   self.gradInput = currentGradOutput
   self.step.decoder = ds + 1
end