local EncDecSearch, EncDec = torch.class('rnn.EncDecSearch', 'rnn.EncDec')

function EncDecSearch:updateOutput(input)
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
      return enc:forward(input)
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

      local prev_enc = self.encoder.clones[ds]
      local nmods = #prev_enc.modules
      local prev_s = prev_enc.modules[nmods].prev_h
      return dec:forward{ input, prev_s }
   end
end

local BiEncDecSearch, BiEncDec = torch.class('rnn.BiEncDecSearch', 'rnn.BiEncDec')

function BiEncDecSearch:updateOutput()
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
      local rv = enc:forward(input)
      self.annotations[{es, {}, {1, self.hiddenSize}}]
         :copy(enc.modules[self.nencmods].prev_h)

      -- Build backward encodings automagically
      for i = es, 1, -1 do
         local input = self.inputs[es]
         local enc_i = self.encoder.clones[i]
         local output = enc_i:forward(input)
         self.annotations[{es, {}, {self.hiddenSize + 1, self.hiddenSize * 2}}]
            :copy(enc_i.modules[self.nencmods].prev_h)
         self.prev = self.prev:typeAs(output):resizeAs(output):copy(output)
      end
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
      self.annotations[{es, {}, {1, self.hiddenSize}}]
         :copy(enc.modules[self.nencmods].prev_h)
      return output
   elseif ds < self.dseqSize then
      if not self.train then
         self.step.decoder = ds + 1
      end

      local prev_enc = self.encoder.clones[ds]
      return dec:forward{ input, self.annotations }
   end
end

function BiEncDec:restart()
   --[[
      EFFECTS:
         Reloads the model to initial values
   ]]

   self.estop = false
   self.step = {
      encoder = 1,
      decoder = 1
   }

   self.prev = torch.Tensor()
   self.inputs = torch.zeros(self.seqSize, self.batchSize, self.inputSize)

   self.nencmods = #self.encoder.clones[1].modules
   self.annotations = torch.zeros(self.seqSize,
      self.batchSize, self.hiddenSize * 2)
end