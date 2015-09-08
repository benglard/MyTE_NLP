local DeepQStack, DeepQ = torch.class('rl.DeepQStack', 'rl.DeepQ')
local RLModule = torch.getmetatable('rl.Module')

function DeepQStack:__init(env, options)
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

         Instead of experience replay, a RNN stack machine is used
         to store "memories".
   ]]

   RLModule.__init(self, env, options)
   self.gamma     = self.options.gamma     or 0.75  -- future reward discount factor
   self.epsilon   = self.options.epsilon   or 0.1   -- for epsilon-greedy policy
   self.lr        = self.options.lr        or 0.01  -- network learning rate
   self.momentum  = self.options.momentum  or 0.9   -- network momentum (only sgd)
   self.hidden    = self.options.hidden    or 100   -- # hidden units
   self.gradclip  = self.options.gradclip  or 1     -- gradient clipping level
   self.usestate  = self.options.usestate  or false -- Use whole state tensor?
   self.nstacks   = self.options.nstacks   or 1     -- # stacks in stack machine
   self.discrete  = self.options.discrete  or false -- Discretize stacks
   self.usenoop   = self.options.usenoop   or false -- Stack with NO-OP
   self.deque     = self.options.deque     or false -- Use rnn.Deque
   self.network   = self.options.network   or self:buildNetwork()
   self.optim     = self.options.optim     or 'sgd'

   self.updates = { self.lr }
   if self.optim == 'sgd' then
      table.insert(self.updates, self.momentum)
   else
      table.insert(self.updates, self.gradclip)
   end
   
   self.gradInput:resize(self.nstates):zero()
   self.ps, self.gs = self.network:getParameters()
end

function DeepQStack:buildNetwork()
   local layer = nil
   if self.deque then layer = rnn.Deque
   else layer = rnn.Stack end

   local model = nn.Sequential()
   model:add(nn.Reshape(1, self.nstates))
   model:add(layer(
      self.nstates, self.hidden,
      self.hidden, 2, self.nstacks,
      self.discrete, self.usenoop))
   model:add(nn.Linear(self.hidden, self.nactions))
   return model
end

function DeepQStack:act(state)
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
      output = torch.zeros(self.nactions)
      output[action] = 1.0
   else
      -- greedy wrt Q function
      output = self.network:forward(input):double()
      output:resize(self.nactions)
      local max, argmax = output:max(1)
      action = argmax:squeeze()
   end

   if self.gpu then
      output = self:transfer(output)
   end

   self.output:resizeAs(output):copy(output)
   self:push(state, action)
   return self.output
end

function DeepQStack:learn(reward)
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
      self.nsteps = self.nsteps + 1
   end
   self.prev_r = reward
end

function DeepQStack:qUpdate()
   --[[
      REQUIRES:
      EFFECTS:
         Computes the value of the Q function given
         the params passed as input, backwards the
         loss through the network, and updates
         the network's paramaters.
   ]]

   local prev_s = self.prev_s
   local prev_a = self.prev_a
   local prev_r = self.prev_r
   local next_s = self.next_s
   local next_a = self.next_a

   -- Compute Q(s, a) = r + gamma * max_a' Q(s', a')
   local input

   if self.usestate then
      input = next_s
   else
      input = self:transfer(torch.zeros(self.nstates))
      input[next_s] = 1.0
   end

   local output = self.network:forward(input)
   local maxQ = prev_r + self.gamma * output:max()

   if self.usestate then
      input = prev_s
   else
      input = self:transfer(torch.zeros(self.nstates))
      input[prev_s] = 1.0
   end
   local pred = self.network:forward(input):resize(self.nactions)

   local loss = pred[prev_a] - maxQ
   local grad = self:transfer(torch.zeros(1, self.nactions))
   grad[1][prev_a] = loss
   grad:clamp(-self.gradclip, self.gradclip)

   local grad = self.network:backward(input, grad)
   self.gradInput:copy(grad)
   self:update(self.network, self.optim, self.updates)
   
   return loss
end