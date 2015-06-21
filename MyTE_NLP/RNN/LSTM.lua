local LSTM, parent = torch.class('rnn.LSTM', 'rnn.Module')

function LSTM:__init(input, hidden, batch, annotate)
   --[[
      REQUIRES:
         input -> a number, size of input
         hidden -> a number, of hidden units to use
         batch -> a number, batch size
         annotate -> a boolean, if true, annotates the
            nodes of the nngraph with local variable names
      EFFECTS:
         Creates an instance of the rnn.LSTM class
         for use in building recurrent neural network
         architectures with LSTM units. 
   ]]

   parent.__init(self, input, hidden, batch)

   self.prev_c = torch.zeros(self.batchSize, self.hiddenSize)
   self.dprev_c = torch.zeros(self.batchSize, self.hiddenSize)
   self.prev_h = torch.zeros(self.batchSize, self.hiddenSize)
   self.dprev_h = torch.zeros(self.batchSize, self.hiddenSize)

   local x = nn.Identity()()
   local prev_c = nn.Identity()()
   local prev_h = nn.Identity()()

   -- Calculate all four gates in one go
   local input = nn.Linear(self.inputSize, self.hiddenSize)(x)
   local i2h   = nn.Linear(self.hiddenSize, 4 * self.hiddenSize)(input)
   local h2h   = nn.Linear(self.hiddenSize, 4 * self.hiddenSize)(prev_h)
   local gates = nn.CAddTable(){ i2h, h2h }

   -- Reshape to (batch_size, n_gates, hid_size)
   -- Then slize the n_gates dimension, i.e dimension 2
   local reshaped = nn.Reshape(4, self.hiddenSize)(gates)
   local sliced   = nn.SplitTable(2)(reshaped)

   -- Use select gate to fetch each gate and apply nonlinearity
   local in_gate      = nn.Sigmoid()(nn.SelectTable(1)(sliced))
   local in_transform = nn.Tanh()(nn.SelectTable(2)(sliced))
   local forget_gate  = nn.Sigmoid()(nn.SelectTable(3)(sliced))
   local out_gate     = nn.Sigmoid()(nn.SelectTable(4)(sliced))

   local memory = nn.CMulTable(){ forget_gate, prev_c }
   local write  = nn.CMulTable(){ in_gate, in_transform }
   local next_c = nn.CAddTable(){ memory, write }
   local next_h = nn.CMulTable(){ out_gate, nn.Tanh()(next_c) }
   local state  = nn.Identity(){ next_c, next_h }

   if annotate then nngraph.annotateNodes() end
   --self.layer = nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
   self.layer = nn.gModule({x, prev_c, prev_h}, {state})
end

function LSTM:updateOutput(input)
   --[[
      REQUIRES:
         input -> a torch Tensor or table
      EFFECTS:
         Feeds input through either the network
         or it's clone at the correct time-step
   ]]

   if type(input) == 'table' then
      self.input = input
   else
      self.input = {input, self.prev_c, self.prev_h}
   end

   local layer = self.clones[self.step] or self.layer
   local next_c, next_h = unpack(layer:updateOutput(self.input))
   self.output:resizeAs(next_h):copy(next_h)
   self.prev_c:resizeAs(next_c):copy(next_c)
   self.prev_h:resizeAs(next_h):copy(next_h)
   return self.output
end

function LSTM:updateGradInput(input, gradOutput)
   --[[
      REQUIRES:
         input -> a torch Tensor
         gradOutput -> a torch Tensor, output of a previous layer
      EFFECTS:
         Calculates the gradient with respect to the
         input to the layer or it's clone at the 
         correct time-step
   ]]

   local gradOutputTable
   if type(gradOutput) == 'table' then
      gradOutputTable = gradOutput
   else
      gradOutputTable = {gradOutput, self.dprev_c, self.dprev_h}
   end

   local layer = self.clones[self.step] or self.layer
   local gix, gic, gih = unpack(layer:updateGradInput(self.input, gradOutputTable))
   self.gradInput:resizeAs(gix):copy(gix)
   self.dprev_c:resizeAs(gic):copy(gic)
   self.dprev_h:resizeAs(gih):copy(gih)
   return self.gradInput
end

function LSTM:accGradParameters(input, gradOutput, scale)
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

   local gradOutputTable
   if type(gradOutput) == 'table' then
      gradOutputTable = gradOutput
   else
      gradOutputTable = {gradOutput, self.dprev_c, self.dprev_h}
   end

   local layer = self.clones[self.step] or self.layer
   layer:accGradParameters(self.input, gradOutputTable)
end