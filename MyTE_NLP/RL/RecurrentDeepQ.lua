local RecurrentDeepQ, DeepQ = torch.class('rl.RecurrentDeepQ', 'rl.DeepQ')
local RLModule = torch.getmetatable('rl.Module')

function RecurrentDeepQ:__init(env, options)
   --[[
      REQUIRES:
         env -> a lua table with keys nstates and nactions,
            and values corresponding to the respective
            number of states and possible actions in the
            agent's environment
         options -> a lua table or nil
      EFFECTS:
         Creates an instance of rl.RecurrentDeepQ, a reinforcement 
         learning agent that approximates the Q value function using a
         multi-layer neural network to maximize future reward.

         Instead of experience replay, a recurrent neural network is used
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
   self.rnntype   = self.options.rnntype   or 'rec' -- RNN type (rec | gru)
   self.nlayers   = self.options.nlayers   or 1     -- # recurrent layers
   self.attend    = self.options.attend    or false -- Use RecurrentAttention
   self.seq       = self.options.seq       or 1     -- RecurrentAttention seq length
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

function RecurrentDeepQ:buildNetwork()
   local model = nn.Sequential()

   if self.attend then
      model:add(nn.Linear(self.nstates, self.hidden))
      model:add(rnn.RecurrentAttention(
         rnn.FFAttention(self.hidden),
         rnn.Recurrent(self.hidden, self.hidden),
         self.seq))
      model:add(nn.Linear(self.hidden, self.nactions))
   else
      local layer
      if     self.rnntype == 'rec'  then layer = rnn.Recurrent
      elseif self.rnntype == 'gru'  then layer = rnn.GRU
      else error('Unsupported layer type: ' .. self.rnntype) end

      model:add(layer(self.nstates, self.hidden))

      for i = 2, self.nlayers do
         model:add(layer(self.hidden, self.hidden))
      end

      model:add(nn.Linear(self.hidden, self.nactions))
   end
   return model
end

function RecurrentDeepQ:act(state)
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

function RecurrentDeepQ:learn(reward)
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

function RecurrentDeepQ:qUpdate()
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

   local grad = self.network:backward(input, grad)
   self.gradInput:copy(grad)
   self:update(self.network, self.optim, self.updates)
   
   return loss
end