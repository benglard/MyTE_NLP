local Recurrent, parent = torch.class('rnn.Recurrent', 'rnn.Module')

function Recurrent:__init(input, hidden, batch, annotate)
   --[[
      REQUIRES:
         input -> a number, size of input
         hidden -> a number, of hidden units to use
         batch -> a number, batch size
         annotate -> a boolean, if true, annotates the
            nodes of the nngraph with local variable names
      EFFECTS:
         Creates an instance of the rnn.Recurrent class
         for use in building neural network architectures
         with recurrent layers. 
   ]]

   parent.__init(self, input, hidden, batch)
   self.prev_h = torch.zeros(self.batchSize, self.hiddenSize)
   self.dprev_h = torch.zeros(self.batchSize, self.hiddenSize)

   local x      = nn.Identity()()
   local prev_h = nn.Identity()()
   local i2h    = nn.Linear(self.inputSize, self.hiddenSize)(x)
   local h2h    = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)
   local next_h = nn.Tanh()(nn.CAddTable(){ i2h, h2h })

   if annotate then nngraph.annotateNodes() end
   self.layer = nn.gModule({x, prev_h}, {next_h})
end

function Recurrent:updateOutput(input)
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
      self.input = {input, self.prev_h}
   end

   local layer = self.clones[self.step] or self.layer
   local next_h = layer:updateOutput(self.input)
   self.output:resizeAs(next_h):copy(next_h)
   self.prev_h:resizeAs(next_h):copy(next_h)
   return self.output
end

function Recurrent:updateGradInput(input, gradOutput)
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
   local gix, gih = unpack(layer:updateGradInput(self.input, gradOutput))
   self.gradInput:resizeAs(gix):copy(gix)
   self.dprev_h:resizeAs(gih):copy(gih)
   return self.gradInput
end

function Recurrent:accGradParameters(input, gradOutput, scale)
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
   layer:accGradParameters(self.input, gradOutput)
end