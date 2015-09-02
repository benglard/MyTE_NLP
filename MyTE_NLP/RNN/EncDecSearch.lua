local EncDecSearch, parent = torch.class('rnn.EncDecSearch', 'rnn.EncDec')

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