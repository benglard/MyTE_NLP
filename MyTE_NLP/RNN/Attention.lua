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
   local a2h = nn.Linear(self.hiddenSize, self.hiddenSize)(annotations)
   local h2h = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)
   local scores = nn.Tanh()(nn.CAddTable(){ a2h, h2h })
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
   local h2h = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)
   local attends = {}

   for j = 1, self.seqSize do
      local h_j = nn.Select(1, j)(annotations)
      local i2h = nn.Linear(self.inputSize, self.hiddenSize)(h_j)
      local scores = nn.Tanh()(nn.CAddTable(){ i2h, h2h })
      local weights = nn.SoftMax()(scores)
      attends[j] = nn.CMulTable(){ weights, annotations }
   end

   local attend = nn.Sum(2)(attends)
   if annotate then nngraph.annotateNodes() end
   self.layer = nn.gModule({annotations, prev_h}, {attend})
end

local RecurrentAttention, _ = torch.class('rnn.RecurrentAttention', 'rnn.Attention')

function RecurrentAttention:__init(attmod, recmod, seq)
   --[[
      REQUIRES:
         attmod -> attention module
         recmod -> recurrent module
         seq -> sequence size
      EFFECTS:
         Creates an instance of the rnn.RecurrentAttention class
         for use in building recurrent neural network architectures
         with soft attention.
   ]]

   Attention.__init(self,
      attmod.inputSize,
      recmod.hiddenSize,
      attmod.batchSize, seq)

   self.amod = nn.Sequential():add(attmod):clone(self.seqSize)
   self.rmod = nn.Sequential():add(recmod):clone(self.seqSize)

   self.layer = nn.Sequential()
      :add(self.rmod)
      :add(self.amod)

   self.fstep = 1
end

function RecurrentAttention:updateOutput(input)
   --[[
      REQUIRES:
         input -> a torch Tensor or table
      EFFECTS:
         Feeds input through either the network
         or it's clone at the correct time-step
   ]]

   local step = self.fstep
   local prev_s
   local prev_rmod = self.rmod.clones[step - 1]
   if prev_rmod == nil then
      prev_s = torch.zeros(self.batchSize, self.inputSize)
   else
      prev_s = prev_rmod.modules[1].prev_h
   end

   local amod = self.amod.clones[step]
   local aout = amod:updateOutput{ input, prev_s }

   local rmod = self.rmod.clones[step]
   local rout = rmod:updateOutput(aout)

   self.output:resizeAs(rout):copy(rout)
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

   local step = self.fstep

   local rmod = self.rmod.clones[step]
   local grad = rmod:updateGradInput(nil, gradOutput)
   self.grad = grad

   local amod = self.amod.clones[step]
   grad = amod:updateGradInput(nil, grad)

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

   local step = self.fstep

   local rmod = self.rmod.clones[step]
   rmod:accGradParameters(nil, gradOutput, scale)
   local amod = self.amod.clones[step]
   amod:updateGradInput(nil, self.grad, scale)

   if step == self.seqSize then
      self.fstep = 1
   else
      self.fstep = step + 1
   end
end

function RecurrentAttention:__tostring__()
   --[[
      EFFECTS:
         Returns the string representation of self.layer
   ]]

   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = string.format('rnn.RecurrentAttention(%d) {%s%s[input',
      #self.amod.clones, line, tab)
   
   str = string.format('%s%s(1)%s(2)%soutput]', str, next, next, next)

   local mod 
   mod = tostring(self.amod.clones[1].modules[1]):gsub(line, line .. tab)
   str = string.format('%s%s%s(1): %s', str, line, tab, mod)
   mod = tostring(self.rmod.clones[1].modules[1]):gsub(line, line .. tab)
   str = string.format('%s%s%s(2): %s', str, line, tab, mod)
   str = str .. line .. '}'

   return str
end