local Deque, Stack = torch.class('rnn.Deque', 'rnn.Stack')
local RNNModule = torch.getmetatable('rnn.Module')

function Deque:__init(input, hidden, p, k, nstacks, discretize, useno, annotate)
   --[[
      REQUIRES:
         input -> a number, size of input
         hidden -> a number, of hidden units to use
         p -> a number, stack size
         k -> a number, of stack elements to consider
         discretize -> a boolean, if true, rounds
            the elements of stack
         useno -> a boolean, if true, includes a NO-OP action
            category, allowing a stack to keep it's top value
         annotate -> a boolean, if true, annotates the
            nodes of the nngraph with local variable names
      EFFECTS:
         Creates an instance of the rnn.Deque class
         for use in building neural network architectures
         with recurrent layers that control a deque machine.
   ]]

   RNNModule.__init(self, input, hidden)
   self.p = p or self.hiddenSize
   self.k = k or 2
   self.nstacks = nstacks or 1

   self.prev_h = torch.zeros(self.batchSize, self.hiddenSize)
   self.dprev_h = torch.zeros(self.batchSize, self.hiddenSize)
   self.dnext_h = torch.zeros(self.batchSize, self.hiddenSize)

   self.prev_s = torch.zeros(self.nstacks, self.p):add(-1)
   self.next_s = torch.zeros(self.nstacks, self.p)
   self.dprev_s = torch.zeros(self.nstacks, self.p)
   self.dnext_s = torch.zeros(self.nstacks, self.p)

   local x      = nn.Identity()()
   local prev_h = nn.Identity()()
   local prev_s = nn.Identity()()

   local nactions = 4
   if useno then nactions = 6 end
   self.nactions = nactions

   local i2h = nn.Linear(self.inputSize, self.hiddenSize)(x)
   local h2h = nn.Linear(self.hiddenSize, self.hiddenSize)(prev_h)

   local stacks = nn.SplitTable(1)(prev_s)
   local next_h_inputs = { i2h, h2h }

   for i = 1, self.nstacks do
      local stack = nn.Reshape(self.p, 1)(nn.SelectTable(i)(stacks))
      local s_k = {}
      for i = 1, self.k do
         s_k[i] = nn.Select(1, i)(stack)
      end
      local s2h = nn.Linear(self.k, self.hiddenSize)(nn.JoinTable(1)(s_k))
      next_h_inputs[i + 2] = s2h
   end

   local next_h = nn.Sigmoid()(nn.CAddTable()(next_h_inputs))
   local outputs = { next_h }

   for i = 1, self.nstacks do
      local stack = nn.Reshape(self.p, 1)(nn.SelectTable(i)(stacks))
      local s_i = {}
      for i = 1, self.p do
         s_i[i] = nn.Select(1, i)(stack)
      end

      local h2a = nn.SoftMax()(nn.Linear(self.hiddenSize, nactions)(next_h))
      local actions = { 
         top = {
            push = nn.Select(2, 1)(h2a),
            pop  = nn.Select(2, 3)(h2a)
         },
         bottom = {
            push = nn.Select(2, 2)(h2a),
            pop  = nn.Select(2, 4)(h2a)
         },
         noop = {}
      }

      local top = nn.Sigmoid()(nn.Linear(self.hiddenSize, 1)(next_h))
      local bottom = nn.Sigmoid()(nn.Linear(self.hiddenSize, 1)(next_h))

      local next_s0_t = {
         nn.CMulTable(){ actions.top.push, top },
         nn.CMulTable(){ actions.top.pop, s_i[2] }
      }
      local next_sp_t = {
         nn.CMulTable(){ actions.bottom.push, bottom },
         nn.CAddTable(){ actions.bottom.pop, s_i[self.p - 1] }
      }

      if useno then
         actions.noop.top = nn.Select(2, 5)(h2a)
         next_s0_t[3] = nn.CMulTable(){ actions.noop.top, s_i[1] }

         actions.noop.bottom = nn.Select(2, 6)(h2a)
         next_sp_t[3] = nn.CMulTable(){ actions.noop.bottom, s_i[self.p] }
      end

      local next_s0 = nn.CAddTable()(next_s0_t)
      local next_sp = nn.CAddTable()(next_sp_t)

      local elements = { next_s0 }
      for i = 2, self.p - 1 do
         elements[i] = nn.CAddTable(){
            nn.CMulTable(){ actions.top.push, s_i[i - 1] },
            nn.CMulTable(){ actions.top.pop,  s_i[i + 1] },
            nn.CMulTable(){ actions.bottom.push, s_i[i + 1] },
            nn.CMulTable(){ actions.bottom.pop,  s_i[i - 1] }
         }
      end
      elements[self.p] = next_sp
      local next_s = nn.JoinTable(1)(elements)

      if discretize then
         next_s = nn.Round()(next_s)
      end

      outputs[i + 1] = next_s
   end

   if annotate then nngraph.annotateNodes() end
   self.layer = nn.gModule({x, prev_h, prev_s}, outputs)
end

function Deque:updateOutput(input)
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
   local outputs = layer:updateOutput(self.input)
   local next_h = outputs[1]
   self.output:resizeAs(next_h):copy(next_h)
   for i = 2, #outputs do
      local next_s = outputs[i]
      self.next_s[{i - 1, {}}]:copy(next_s)
   end
   return self.output
end

function Deque:updateGradInput(input, gradOutput)
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
   self.gradOutputTable = { gradOutput }
   for i = 1, self.nstacks do
      self.gradOutputTable[i + 1] = self.dprev_s[{i, {}}]
   end

   local gradInputs = layer:updateGradInput(self.input, self.gradOutputTable)
   local gix, gih, gis = unpack(gradInputs)
   self.gradInput:resizeAs(gix):copy(gix)
   self.dnext_h:resizeAs(gih):copy(gih)
   self.dnext_s:resizeAs(gis):copy(gis)
   return self.gradInput
end