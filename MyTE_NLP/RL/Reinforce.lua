local Reinforce, parent = torch.class('rl.Reinforce', 'rl.Module')

function Reinforce:__init(env, options)
   --[[
      REQUIRES:
         env -> a lua table with keys nstates and nactions,
            and values corresponding to the respective
            number of states and possible actions in the
            agent's environment
         options -> a lua table or nil
      EFFECTS:
         Creates an instance of rl.Reinforce, a reinforcement learning
         agent that approximates the value function using a
         multi-layer neural network to maximize future reward.
   ]]

   parent.__init(self, env, options)
   self.gamma     = self.options.gamma     or 0.75  -- future reward discount factor
   self.epsilon   = self.options.epsilon   or 0.1   -- for epsilon-greedy policy
   self.lr        = self.options.lr        or 0.01  -- network learning rate
   self.baselr    = self.options.baselr    or 0.01  -- baseline network learning rate
   self.hidden    = self.options.hidden    or 100   -- # hidden units
   self.bhidden   = self.options.bhidden   or 100   -- # baseline hidden units
   self.variance  = self.options.variance  or 0.02  -- for stochastic gaussian policy
   self.gradclip  = self.options.gradclip  or 0.1   -- gradient clipping level
   self.momentum  = self.options.momentum  or 0.9   -- network momentum (only sgd)
   self.batchsize = self.options.batchsize or 100   -- train every n instances
   self.history   = self.options.history   or 80    -- use this many past instances to compute actual rewards
   self.rectifier = self.options.rectifier or nn.Tanh
   self.network   = self.options.network   or self:buildNetwork()
   self.baseline  = self.options.baseline  or self:buildBaseline()
   self.optim     = self.options.optim     or 'sgd'

   self.updates = { self.lr }
   if self.optim == 'sgd' then
      table.insert(self.updates, self.momentum)
   else
      table.insert(self.updates, self.gradclip)
   end
   
   self.states = {}
   self.rewards = {}
   self.actions = {}
   self.outputs = {
      network = {},
      baseline = {}
   }
   self.grads = {
      network = -1,
      baseline = -1,
   }
end

function Reinforce:_build(hidden)
   --[[
      REQUIRES:
         hidden -> number of hidden units
      EFFECTS:
         Returns the underlying network used to
         train the Reinforce agent.
   ]]

   local model = nn.Sequential()
   model:add(nn.Linear(self.nstates, hidden))
   model:add(self.rectifier())
   model:add(nn.Linear(hidden, hidden))
   model:add(self.rectifier())
   model:add(nn.Linear(hidden, self.nactions))
   return model
end

function Reinforce:buildNetwork()
   --[[
      EFFECTS:
         Returns the underlying network used to
         train the Reinforce agent.
   ]]

   return self:_build(self.hidden)
end

function Reinforce:buildBaseline()
   --[[
      EFFECTS:
         Returns the underlying network used to
         train the Reinforce agent baseline.
   ]]

   return self:_build(self.bhidden)
end

function Reinforce:act(state)
   --[[
      REQUIRES:
         state -> a one-hot torch Tensor of size self.nstates
      EFFECTS:
         Perform an "action" on the state of the agent's
         environment by forwarding the current state
         through the network. Also forwards the current
         state through the baseline network to compute
         the baseline.
   ]]

   table.insert(self.states, state)

   -- Forward state through network
   local output = self.network:forward(state):clone()
   table.insert(self.outputs.network, output)

   -- Forward state through baseline
   local base = self.baseline:forward(state):clone()
   table.insert(self.outputs.baseline, base)

   -- Sample an action
   local action = torch.Tensor(self.nactions):copy(output)
   action[1] = torch.normal(0, self.variance) + action[1]
   action[2] = torch.normal(0, self.variance) + action[2]   
   table.insert(self.actions, action)

   self.output:resizeAs(action):copy(action)
   return self.output
end

function Reinforce:learn(reward)
   --[[
      REQUIRES:
         reward -> a number representing the agent's
         reward after performing an action
      EFFECTS:
         Computes the discounted reward, computes the
         gradient of the network's input and the gradient
         of the network baselines's input, passes the
         gradients through the respective networks,
         and updates both networks.
   ]]

   table.insert(self.rewards, reward)
   local n = #self.rewards
   local baseMSE = 0.0
   local experience = self.batchsize
   local age = self.history

   if n >= experience then -- Learn
      for t = 1, age do
         -- Actual discounted reward for this time step
         local mul = 1
         local reward = 0
         for t2 = t, n do
            reward = reward + mul * self.rewards[t2]
            mul = mul * self.gamma
            if mul < 1e-5 then break end
         end

         -- Predicted baseline at this time step
         local action = self.actions[t]
         local max, argmax = action:max(1)
         local idx = argmax:squeeze()

         local base = self.outputs.baseline[t][idx] - reward
         local update = torch.add(self.actions[t], -self.outputs.network[t])
         update:mul(base)
         update:clamp(-self.gradclip, self.gradclip)
         self.grads.network = torch.zeros(self.nactions):copy(update)

         update = torch.Tensor{base}:clamp(-self.gradclip, self.gradclip):squeeze()
         self.grads.baseline = torch.zeros(self.nactions)
         self.grads.baseline[idx] = update

         baseMSE = baseMSE + (update * update)

         self.network:backward(self.states[t], self.grads.network)
         self.network:backward(self.states[t], self.grads.baseline)
      end

      -- update network
      self:update(self.network, self.optim, self.updates)
      self:update(self.baseline, self.optim, self.updates)
      self.loss = baseMSE / age

      -- reset
      self.states = {}
      self.rewards = {}
      self.actions = {}
      self.outputs.network = {}
      self.outputs.baseline = {}
      self.grads.network = -1
      self.grads.baseline = -1
   end

   self.nsteps = self.nsteps + 1
   self.prev_r = reward
end