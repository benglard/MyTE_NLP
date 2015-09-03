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
      local h_j = prev_enc.modules[nmods].prev_h

      local prev_dec = self.decoder.clones[ds - 1]
      local prev_s
      if prev_dec == nil then
         prev_s = torch.zeros(self.batchSize, self.hiddenSize)
      else
         prev_s = prev_dec.modules[2].prev_h
      end

      local attended = dec.modules[1]:forward{ h_j, prev_s }
      dec.modules[2].prev_h:copy(attended)
      local output = input
      for i = 2, #dec.modules do
         output = dec.modules[i]:forward(output)
      end
      return output
   end
end

function EncDecSearch:state()
   --[[
      EFFECTS:
         Returns hidden state of first layer
         of decoder
   ]]

   local ds = self.step.decoder - 1
   return self.decoder.clones[ds].modules[2].output:clone()
end

local BiEncDecSearch, BiEncDec = torch.class('rnn.BiEncDecSearch', 'rnn.BiEncDec')

function BiEncDecSearch:updateOutput(input)
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

      local hs = self.hiddenSize
      self.annotations[{es, {}, {1, hs}}]
         :copy(enc.modules[self.nencmods].prev_h)

      -- Build backward encodings automagically
      for i = es, 2, -1 do
         local input = self.inputs[es]:resize(self.inputSize)
         local enc_i = self.encoder.clones[i]
         local output = enc_i:forward(input)
         self.annotations[{es, {}, {self.hiddenSize + 1, self.hiddenSize * 2}}]
            :copy(enc_i.modules[self.nencmods].prev_h)
         self.prev = self.prev:typeAs(output):resizeAs(output):copy(output)
      end

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
      self.annotations[{es, {}, {1, self.hiddenSize}}]
         :copy(enc.modules[self.nencmods].prev_h)
      return output
   elseif ds < self.dseqSize then
      if not self.train then
         self.step.decoder = ds + 1
      end

      local prev_dec = self.decoder.clones[ds - 1]
      local h_t
      if prev_dec == nil then
         h_t = torch.zeros(self.batchSize, self.hiddenSize * 2)
      else
         h_t = prev_dec.modules[2].prev_h
      end

      local attended = dec.modules[1]:forward{ self.annotations[ds], h_t }
      dec.modules[2].prev_h:copy(attended)
      local output = input
      for i = 2, #dec.modules do
         output = dec.modules[i]:forward(output)
      end
      return output
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

function BiEncDecSearch:state()
   --[[
      EFFECTS:
         Returns hidden state of first layer
         of decoder
   ]]

   local ds = self.step.decoder - 1
   return self.decoder.clones[ds].modules[2].output:clone()
end