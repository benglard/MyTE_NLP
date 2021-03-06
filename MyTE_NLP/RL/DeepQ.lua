-------- DeepQMemory

local Memory = torch.class('rl.DeepQMemory')

function Memory:__init(size)
   --[[
      REQUIRES:
         size -> the size of a DeepQMemory module
      EFFECTS:
         Creates an instance of rl.DeepQMemory
         for use in storing and replaying a
         DeepQ agent's experiences.
   ]]

   self.size = size
   self.tape = {}
   self.pos = 1
end

function Memory:append(state)
   --[[
      REQUIRES:
         state -> a lua table storing the state
         or experience of a DeepQ agent
      EFFECTS:
         Places state on the memory tape
         at it's current position. Increments
         the tape's position and resets if
         memory is full.
   ]]

   self.tape[self.pos] = state
   self.pos = self.pos + 1
   if self.pos > self.size then
      self:clear()
   end
end

function Memory:clear()
   --[[
      EFFECTS:
         Resets the memory tape's position
         to 1
   ]]

   self.pos = 1
end

function Memory:sample()
   --[[
      EFFECTS:
         Samples an experience from memory
         for replay 
   ]]

   local idx = torch.random(1, #self.tape)
   return self.tape[idx]
end

-------- DeepQ

local DeepQ, parent = torch.class('rl.DeepQ', 'rl.Module')

function DeepQ:__init(env, options)
   --[[
      REQUIRES:
         env -> a lua table with keys nstates and nactions,
            and values corresponding to the respective
            number of states and possible actions in the
            agent's environment
         options -> a lua table or nil
      EFFECTS:
         Creates an instance of rl.DeepQ, a reinforcement learning
         agent that approximates the Q value function using a
         multi-layer neural network to maximize future reward.
   ]]

   parent.__init(self, env, options)
   self.gamma     = self.options.gamma     or 0.75  -- future reward discount factor
   self.epsilon   = self.options.epsilon   or 0.1   -- for epsilon-greedy policy
   self.lr        = self.options.lr        or 0.01  -- network learning rate
   self.momentum  = self.options.momentum  or 0.9   -- network momentum (only sgd)
   self.hidden    = self.options.hidden    or 100   -- # hidden units
   self.memory    = self.options.memory    or 5000  -- size of experience replay
   self.batchsize = self.options.batchsize or 10    -- # time steps to sample and learn from
   self.gradclip  = self.options.gradclip  or 1     -- gradient clipping level
   self.usestate  = self.options.usestate  or false -- Use whole state tensor?
   self.nlayers   = self.options.nlayers   or 1     -- # layers
   self.rectifier = self.options.rectifier or nn.Tanh
   self.network   = self.options.network   or self:buildNetwork()
   self.optim     = self.options.optim     or 'sgd'

   self.updates = { self.lr, learningRate = self.lr }
   if self.optim == 'sgd' then
      table.insert(self.updates, self.momentum)
      self.updates.momentum = self.momentum
   else
      table.insert(self.updates, self.gradclip)
   end
   
   self.memory = rl.DeepQMemory(self.memory)
   self.gradInput:resize(self.nstates):zero()
   self.ps, self.gs = self.network:getParameters()
end

function DeepQ:buildNetwork()
   --[[
      EFFECTS:
         Returns the underlying network used to
         train the DeepQ agent.
   ]]

   local model = nn.Sequential()
   model:add(nn.Linear(self.nstates, self.hidden))
   model:add(self.rectifier())
   for i = 1, self.nlayers do
      model:add(nn.Linear(self.hidden, self.hidden))
      model:add(self.rectifier())
   end
   model:add(nn.Linear(self.hidden, self.nactions))
   return model
end

function DeepQ:act(state)
   --[[
      REQUIRES:
         state -> a number between 1 and self.nstates
         or a one-hot torch Tensor of size self.nstates
      EFFECTS:
         Perform an "action" on the state of the agent's
         environment by forwarding the current state
         through the network.
   ]]

   local action = -1
   local input, output

   if type(state) == 'number' then
      assert(state > 0 and state <= self.nstates)
      input = torch.zeros(self.nstates)
      input[state] = 1.0
   else
      input = state:clone()
      if not self.usestate then
         local max, argmax = input:max(1)
         state = argmax:squeeze()
      end
   end
   input = self:transfer(input)

   if math.random() < self.epsilon then
      -- epsilon greedy policy
      action = torch.random(1, self.nactions)
      output = self:transfer(torch.zeros(self.nactions))
      output[action] = 1.0
   else
      -- greedy wrt Q function
      output = self.network:forward(input)
      local max, argmax = output:max(1)
      action = argmax:squeeze()
   end

   self.output:resizeAs(output):copy(output)
   self:push(state, action)
   return self.output
end

function DeepQ:learn(reward)
   --[[
      REQUIRES:
         reward -> a number representing the agent's
         reward after performing an action
      EFFECTS:
         Performs a qUpdate using the agent's current
         state, stores the experience in replay memory
         (if on interval), and samples some old experiences
         from memory and performs a qUpdate on those
         experiences.
   ]]

   if self.prev_r ~= nil and self.lr > 0 then
      -- qUpdate, loss is a measure of surprise to the agent
      self.loss = self:qUpdate()

      -- Store experience in replay memory
      local exp = {
         self.prev_s, self.prev_a, self.prev_r,
         self.next_s, self.next_a
      }
      self.memory:append(exp)
      self.nsteps = self.nsteps + 1
      
      -- Sample some additional experience from replay memory and learn from it
      if self.nsteps % self.batchsize == 0 then
         for i = 1, self.batchsize do
            local exp = self.memory:sample()
            self:qUpdate(unpack(exp))
         end
         self.memory:clear()
      end
   end
   self.prev_r = reward
end

function DeepQ:qUpdate(prev_s, prev_a, prev_r, next_s, next_a)
   --[[
      REQUIRES:
         prev_s -> previous state
         prev_a -> previous action
         prev_r -> previous reward
         next_s -> next state
         next_a -> next reward
      EFFECTS:
         Computes the value of the Q function given
         the params passed as input, backwards the
         loss through the network, and updates
         the network's paramaters.
   ]]

   local learned = false
   if prev_s == nil then
      prev_s = self.prev_s
      learned = true
   end
   prev_a = prev_a or self.prev_a
   prev_r = prev_r or self.prev_r
   next_s = next_s or self.next_s
   next_a = next_a or self.next_a

   -- Compute Q(s, a) = r + gamma * max_a' Q(s', a')
   local input
   if self.usestate then
      input = self:transfer(next_s)
   else
      input = self:transfer(torch.zeros(self.nstates))
      input[next_s] = 1.0
   end

   local output = self.network:forward(input)
   local maxQ = prev_r + self.gamma * output:max()

   if self.usestate then
      input = self:transfer(prev_s)
   else
      input = self:transfer(torch.zeros(self.nstates))
      input[prev_s] = 1.0
   end

   local pred = self.network:forward(input)
   local loss = pred[prev_a] - maxQ
   local grad = self:transfer(torch.zeros(self.nactions))
   grad[prev_a] = loss
   grad:clamp(-self.gradclip, self.gradclip)

   local currentGradInput = self.network:backward(input, grad)
   if learned then
      self.gradInput:copy(currentGradInput)
      self:update(self.network, self.optim, self.updates, loss)
   end
   return loss
end

function DeepQ:cuda()
   if cutorch ~= nil then
      self.gpu = true
      self.network = self.network:cuda():clone()
      self.w = self.w:cuda():clone()
      self.dw = self.dw:cuda():clone()
      self.ps, self.gs = self.network:getParameters()
      self.output = self.output:cuda()
      self.gradInput = self.gradInput:cuda()
   end
   return self
end

function DeepQ:transfer(v)
   if self.gpu then
      return v:cuda()
   else
      return v:double()
   end
end

function DeepQ:sgd(net, lr, momentum)
   --[[
      REQUIRES:
         net -> an instance of nn
         lr -> network learning rate, a number or nil
         momentum -> network momentum, a number or nil
      EFFECTS:
         Updates net using stochastic gradient
         descent.
   ]]

   lr = lr or 0.01
   local mom = momentum or 0.9

   local ps, gs = self.ps, self.gs
   gs:mul(-lr)

   if mom > 0.0 then
      self.dw:resizeAs(gs):mul(mom):add(gs)
      ps:add(self.dw)
   else
      ps:add(gs)
   end

   gs:zero()
end

function DeepQ:rmsprop(net, lr, clip)
   --[[
      REQUIRES:
         net -> an instance of nn
         lr -> network learning rate, a number or nil
         momentum -> gradient clipping level, a number or nil
      EFFECTS:
         Updates net using the rmsprop optmization
         method.
   ]]

   lr = lr or 0.01
   clip = clip or 5

   local epsilon = 1e-8
   local decay = 0.999
   local reg = 0.0001

   local ps, gs = self.ps, self.gs
   self.w:resizeAs(gs)
   self.dw:resizeAs(gs)

   -- update cache
   self.w:mul(decay):add(torch.cmul(gs, gs):mul(1 - decay))

   -- clip gradients
   gs:clamp(-clip, clip)

   -- update params
   self.dw
      :mul(gs, -lr)
      :cdiv(torch.add(self.w, epsilon):sqrt())
      :add(torch.mul(ps, -reg))
   ps:add(self.dw)

   -- zero gradients
   gs:zero()
end

function DeepQ:save(path, forgpu)
   --[[
      REQUIRES:
         path -> path to save model to
         forgpu -> save gpu or cpu model, defaults to self.gpu
      EFFECTS:
         Saves a DeepQ model to disk
   ]]

   if forgpu == nil then
      forgpu = self.gpu
   end

   if forgpu and self.gpu then
      path = path .. '.gpu'
      torch.save(path, {
         network = self.network,
         w = self.w,
         dw = self.dw
      })
   else
      path = path .. '.cpu'
      torch.save(path, {
         network = self.network:double(),
         w = self.w:double(),
         dw = self.dw:double()
      })
   end
end

function DeepQ:load(path)
   --[[
      REQUIRES:
         path -> path to load from
      EFFECTS:
         Loads a saved DeepQ model
   ]]

   local data = torch.load(path)
   self.network = data.network
   self.ps, self.gs = self.network:getParameters()
   self.w = data.w
   self.dw = data.dw
end