local Attention, Module = torch.class('rnn.Attention', 'rnn.Module')

function Attention:updateOutput(input)   
   --[[
      REQUIRES:
         input -> table of annotations and activations
            of hidden layer that produced the annotations
      EFFECTS:
         Outputs the attended input
   ]]

   local layer = self.clones[self.step] or self.layer
   self.input = input
   local next_h = layer:updateOutput(self.input)
   self.output:resizeAs(next_h):copy(next_h)
   return self.output
end

function Attention:updateGradInput(input, gradOutput)
   --[[
      REQUIRES:
         input -> a torch Tensor
         gradOutput -> a torch Tensor, output of a previous layer
      EFFECTS:
         Calculates the gradient with respect to the
         input to the layer or it's clone at the 
         correct time-step
   ]]

   local layer = self.clones[self.step] or self.layer
   local gix, _ = unpack(layer:updateGradInput(self.input, gradOutput))
   self.gradInput:resizeAs(gix):copy(gix)
   return self.gradInput
end

function Attention:accGradParameters(input, gradOutput, scale)
   --[[
      REQUIRES:
         input -> a torch Tensor or table
         gradOutput -> a torch Tensor, output of a previous layer
         scale -> a number
      EFFECTS:
         Calculates the gradient with respect to the
         parameters of the layer or it's clone at the 
         correct time-step
   ]]

   local layer = self.clones[self.step] or self.layer
   layer:accGradParameters(self.input, gradOutput, scale)
end

local FFAttention, _ = torch.class('rnn.FFAttention', 'rnn.Attention')

function FFAttention:__init(hidden, batch, annotate)
   --[[
      REQUIRES:
         hidden -> number of hidden units
         batch -> size of batch
         annotate -> boolean, whether to annotate nngraph nodes
      EFFECTS:
         Creates an instance of the rnn.FFAttention class
         for use in building recurrent neural network architectures
         with soft attention using simple feed-forward neural nets.
   ]]

   Attention.__init(self, hidden, hidden, batch)

   local annotations = nn.Identity()()
   local prev_h = nn.Identity()()
   local i2h = nn.Linear(self.inputSize, self.hiddenSize)(annotations)
   local h2h = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)
   local scores = nn.Tanh()(nn.CAddTable(){ i2h, h2h })
   local weights = nn.SoftMax()(scores)
   local attend = nn.CMulTable(){ weights, annotations }

   if annotate then nngraph.annotateNodes() end
   self.layer = nn.gModule({annotations, prev_h}, {attend})
end

local SequenceAttention, _ = torch.class('rnn.SequenceAttention', 'rnn.Attention')

function SequenceAttention:__init(input, hidden, batch, annotate)
   --[[
      REQUIRES:
         input -> number, size of input
         hidden -> number of hidden units
         batch -> size of batch
         annotate -> boolean, whether to annotate nngraph nodes
      EFFECTS:
         Creates an instance of the rnn.SequenceAttention class
         for use in building recurrent neural network architectures
         with soft attention over sequences.
   ]]

   Attention.__init(self, input, hidden, batch)

   local annotations = nn.Identity()()
   local prev_h = nn.Identity()()
   local i2h = nn.Linear(self.inputSize, self.hiddenSize)(annotations)
   local h2h = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)
   local scores = nn.Tanh()(nn.CAddTable(){ i2h, h2h })
   local weights = nn.SoftMax()(scores)
   local attend = nn.CMulTable(){ weights, annotations }

   if annotate then nngraph.annotateNodes() end
   self.layer = nn.gModule({annotations, prev_h}, {attend})
end

local RecurrentAttention, _ = torch.class('rnn.RecurrentAttention', 'rnn.Attention')

function RecurrentAttention:__init(recmod, attmod, seq)
   --[[
      REQUIRES:
         recmod -> recurrent module
         attmod -> attention module
         seq -> sequence size
      EFFECTS:
         Creates an instance of the rnn.RecurrentAttention class
         for use in building recurrent neural network architectures
         with soft attention.
   ]]

   Attention.__init(self,
      recmod.inputSize,
      recmod.hiddenSize,
      recmod.batchSize, seq)

   self.rmod = nn.Sequential():add(recmod):clone(self.seqSize)
   self.amod = nn.Sequential():add(attmod):clone(self.seqSize)

   self.layer = nn.Sequential()
      :add(self.rmod)
      :add(self.amod)

   self.step = 1
end

function RecurrentAttention:updateOutput(input)
   --[[
      REQUIRES:
         input -> a torch Tensor or table
      EFFECTS:
         Feeds input through either the network
         or it's clone at the correct time-step
   ]]

   local rmod = self.rmod.clones[self.step]
   local rout = rmod:updateOutput(input)
   local amod = self.amod.clones[self.step]
   local aout = amod:updateOutput{ rout, rmod.modules[1].prev_h }
   self.output:resizeAs(aout):copy(aout)
   return self.output
end

function RecurrentAttention:updateGradInput(input, gradOutput)
   --[[
      REQUIRES:
         input -> a torch Tensor
         gradOutput -> a torch Tensor, output of a previous layer
      EFFECTS:
         Calculates the gradient with respect to the
         input to the layer or it's clone at the 
         correct time-step
   ]]

   local amod = self.amod.clones[self.step]
   local grad = amod:updateGradInput(nil, gradOutput)
   self.grad = grad
   local rmod = self.rmod.clones[self.step]
   grad = rmod:updateGradInput(nil, grad)
   self.gradInput:resizeAs(grad):copy(grad)
   return self.gradInput
end

function RecurrentAttention:accGradParameters(input, gradOutput, scale)
   --[[
      REQUIRES:
         input -> a torch Tensor or table
         gradOutput -> a torch Tensor, output of a previous layer
         scale -> a number
      EFFECTS:
         Calculates the gradient with respect to the
         parameters of the layer or it's clone at the 
         correct time-step
   ]]

   local amod = self.amod.clones[self.step]
   amod:accGradParameters(nil, gradOutput, scale)
   local rmod = self.rmod.clones[self.step]
   rmod:updateGradInput(nil, self.grad, scale)

   if self.step == self.seqSize then
      self.step = 1
   else
      self.step = self.step + 1
   end
end

function RecurrentAttention:__tostring__()
   --[[
      EFFECTS:
         Returns the string representation of self.layer
   ]]

   local template = 'rnn.RecurrentAttention(): %s'
   return string.format(template, self.layer)
end