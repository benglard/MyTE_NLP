local GRU, parent = torch.class('rnn.GRU', 'rnn.Module')

function GRU:__init(input, hidden, batch, annotate)
   --[[
      REQUIRES:
         input -> a number, size of input
         hidden -> a number, of hidden units to use
         batch -> a number, batch size
         annotate -> a boolean, if true, annotates the
            nodes of the nngraph with local variable names
      EFFECTS:
         Creates an instance of the rnn.GRU class
         for use in building recurrent neural network
         architectures with gated recurrent units.
   ]]

   parent.__init(self, input, hidden, batch)
   self.prev_h = torch.zeros(self.batchSize, self.hiddenSize)
   self.dprev_h = torch.zeros(self.batchSize, self.hiddenSize)

   local x = nn.Identity()()
   local prev_h = nn.Identity()()

   local function sum()
      local i2h = nn.Linear(self.inputSize, self.hiddenSize)(x)
      local h2h = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)
      return nn.CAddTable(){ i2h, h2h }
   end

   local update_gate = nn.Sigmoid()(sum())
   local reset_gate  = nn.Sigmoid()(sum())

   -- compute candidate hidden state
   local gated_hidden = nn.CMulTable(){ reset_gate, prev_h }
   local p2 = nn.Linear(self.hiddenSize, self.hiddenSize)(gated_hidden)
   local p1 = nn.Linear(self.inputSize, self.hiddenSize)(x)
   local hidden_candidate = nn.Tanh()(nn.CAddTable(){ p1,p2 })

   -- compute new interpolated hidden state, based on the update gate
   local zh = nn.CMulTable(){ update_gate, hidden_candidate }
   local zhm1 = nn.CMulTable(){
      nn.AddConstant(1, false)(nn.MulConstant(-1, false)(update_gate)),
      prev_h
   }
   local next_h = nn.CAddTable(){ zh, zhm1 }
   
   if annotate then nngraph.annotateNodes() end 
   self.layer = nn.gModule({x, prev_h}, {next_h})
end

function GRU:updateOutput(input)
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

function GRU:updateGradInput(input, gradOutput)
   --[[
      REQUIRES:
         input -> a torch Tensor
         gradOutput -> a torch Tensor, output of a previous layer
      EFFECTS:
         Backpropogates input and gradOutput through 
         either the network or it's clone at the 
         correct time-step
   ]]

   local layer = self.clones[self.step] or self.layer
   local gix, gih = unpack(layer:updateGradInput(self.input, gradOutput))
   self.gradInput:resizeAs(gix):copy(gix)
   self.dprev_h:resizeAs(gih):copy(gih)
   return self.gradInput
end