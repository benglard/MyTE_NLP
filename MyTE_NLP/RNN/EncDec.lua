local EncDec, parent = torch.class('rnn.EncDec', 'rnn.Module')

function EncDec:__init(encoder, decoder, stop, input, hidden, batch, seq, vocab, dseq)
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
         dseq -> a number or nil, nclones of decoder or seq
      EFFECTS:
         Creates an instance of the rnn.EncDec class for use
         in building recurrent nueral network
         encoder-decoder models.
   ]]

   parent.__init(self, input, hidden, batch, seq)
   self.encoder = encoder:clone(self.seqSize)

   local ndc = self.seqSize
   dseq = dseq or false
   if dseq then ndc = dseq end
   self.decoder = decoder:clone(ndc)
   self.dseqSize = ndc

   self.layer = nn.Sequential()
   self.layer:add(encoder)
   self.layer:add(decoder)

   self.stop = stop
   self.vocabSize = vocab
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
      input = torch.Tensor(self.inputSize):typeAs(self.prev):fill(self.vocabSize)
      self.inputs[es]:copy(input)
      self.decoder.clones[1].modules[1].prev_h:copy(self.prev)
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
   local ds = self.step.decoder
   local dec = self.decoder.clones[ds]
   local currentGradOutput = dec:backward(input, gradOutput, scale)
   dec.gradInput = currentGradOutput

   if ds == 1 then
      local es = self.step.encoder
      for i = es, 1, -1 do
         local encinput = self.inputs[i]
         local enc = self.encoder.clones[i]
         enc:backward(encinput, currentGradOutput, scale)
      end
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
   return self.decoder.clones[ds].modules[1].output:clone()
end

function EncDec:restart()
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
end

function EncDec:__tostring__()
   --[[
      EFFECTS:
         Returns the string representation of
         self.layer
   ]]

   local template = '%s(%s,%s): %s'
   return string.format(template, torch.type(self),
      self.seqSize, self.dseqSize, self.layer)
end

EncDec.encode = EncDec.updateOutput