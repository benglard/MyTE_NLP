local Stack, parent = torch.class('rnn.Stack', 'rnn.Module')

function Stack:__init(input, hidden, p, k, annotate)
   --[[
      REQUIRES:
         input -> a number, size of input
         hidden -> a number, of hidden units to use
         p -> a number, stack size
         k -> number of stack elements to consider
         annotate -> a boolean, if true, annotates the
            nodes of the nngraph with local variable names
      EFFECTS:
         Creates an instance of the rnn.Recurrent class
         for use in building neural network architectures
         with recurrent layers. 
   ]]

   parent.__init(self, input, hidden)
   self.p = p or self.hiddenSize
   self.k = k or 2
   
   self.prev_h = torch.zeros(self.batchSize, self.hiddenSize)
   self.dprev_h = torch.zeros(self.batchSize, self.hiddenSize)
   self.dnext_h = torch.zeros(self.batchSize, self.hiddenSize)

   self.prev_s = torch.zeros(self.p):add(-1) -- empty
   self.next_s = torch.zeros(self.p)
   self.dprev_s = torch.zeros(self.p)
   self.dnext_s = torch.zeros(self.p)

   local x      = nn.Identity()()
   local prev_h = nn.Identity()()
   local prev_s = nn.Identity()()
   local inputs = { x, prev_h, prev_s }

   local i2h    = nn.Linear(self.inputSize, self.hiddenSize)(x)
   local h2h    = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)
   local s2h    = nn.Linear(self.hiddenSize, self.p)(prev_s)
   local next_h = nn.Sigmoid()(nn.CAddTable(){ i2h, h2h, s2h })
   local h2a    = nn.SoftMax()(nn.LinearNoBias(self.hiddenSize, 2)(next_h))

   local push  = nn.Select(2, 1)(h2a)
   local pop   = nn.Select(2, 2)(h2a)
   local stack = nn.Reshape(self.p, 1)(prev_s)

   local top = nn.Sigmoid()(nn.LinearNoBias(self.hiddenSize, 1)(next_h))
   local h2s0 = nn.CAddTable(){
      nn.CMulTable(){ push, top },
      nn.CMulTable(){ pop, nn.Select(1, self.p - 1)(stack) }
   }
   local outputs = { h2s0 }
   for i = self.p - 1, 2, -1 do
      local s_im1 = nn.Select(1, i - 1)(stack)
      local s_ip1 = nn.Select(1, i + 1)(stack)
      local h2si = nn.CAddTable(){
         nn.CMulTable(){ push, s_im1 },
         nn.CMulTable(){ pop, s_ip1 }
      }
      table.insert(outputs, h2si)
   end
   local next_s = nn.JoinTable(1)(outputs)

   if annotate then nngraph.annotateNodes() end
   self.layer = nn.gModule(inputs, {next_h, next_s})
end

function Stack:updateOutput(input)
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
      self.input = {input, self.prev_h, self.prev_s}
   end
   
   local layer = self.clones[self.step] or self.layer
   local next_h, next_s = unpack(layer:updateOutput(self.input))
   self.output:resizeAs(next_h):copy(next_h)
   self.next_s[{{1, self.p - 1}}]:copy(next_s)
   return self.output
end

function Stack:updateGradInput(input, gradOutput)
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
   self.gradOutputTable = { gradOutput, self.dprev_s[{{1, self.p - 1}}] }
   local gradInputs = layer:updateGradInput(self.input, self.gradOutputTable)
   local gix, gih, gis = unpack(gradInputs)
   self.gradInput:resizeAs(gix):copy(gix)
   self.dnext_h:resizeAs(gih):copy(gih)
   self.dnext_s:resizeAs(gis):copy(gis)
   return self.gradInput
end

function Stack:accGradParameters(input, gradOutput, scale)
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
   layer:accGradParameters(self.input, self.gradOutputTable, scale)
   self.prev_h:copy(self.output)
   self.prev_s:copy(self.next_s)
   self.dprev_h:copy(self.dnext_h)
   self.dprev_s:copy(self.dnext_s)
end