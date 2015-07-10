local RLTrainer = torch.class('rl.RLTrainer')

function RLTrainer:__init(agent, stateFunc, rewardFunc)
   --[[
      REQUIRES:
         agent -> a reinforcement learning agent
         stateFunc -> function with no inputs, which
            computes and returns the current state
            of the agent
         rewardFunc -> a function with one input
            representing the action the agent has just
            taken, which computes and returns the
            reward of that action (a number).
      EFFECTS:
         Creates an instance of rl.RLTrainer for
         use in training reinforcement learning
         agents.
   ]]

   self.agent = agent
   self.sf = stateFunc
   self.rf = rewardFunc
end

function RLTrainer:train(params)
   --[[
      REQUIRES:
         params -> a lua table or nil
      EFFECTS:
         Trains a reinforcment learning agent
         by continually recieving a state,
         acting on it, recieiving a reward,
         and learning based on it.
   ]]

   params = params or {}
   local verbose  = params.verbose or false
   local epsdecay = params.epsdecay or 1
   local maxR = params.maxR or 1e8
   local minR = params.minR or 1e-8
   local ntrain = params.ntrain or false
   local c = 1

   while true do
      local state = self.sf()
      local action = self.agent:forward(state)
      local reward = self.rf(action)
      if reward > maxR then reward = maxR end
      if reward < minR then reward = minR end
      self.agent:backward(reward)
      self.agent.epsilon = self.agent.epsilon * epsdecay

      if verbose then
         print(string.format('N: %d, Reward: %.3f, Loss: %.3f', 
            self.agent.nsteps, reward, self.agent.loss))
      else
         if ntrain then
            xlua.progress(c, ntrain)
            c = c + 1
            if c > ntrain then c = 1 end
         end
      end
   end
end

function RLTrainer:predict(input, params)
   --[[
      REQUIRES:
         input -> input to self.agent.network.forward
         params -> set of parameters
      EFFECTS:
         Returns the predicted action for
         a specific input
   ]]

   params = params or {}
   local sample = params.sample or false
   local temp = params.temp or 0.1
   local length = params.length or 1

   local rv = {input}
   for n = 1, length do
      local o = self.agent.network:forward(input):clone()
      if sample then
         local probs = torch.exp(o:div(temp))
         probs:div(probs:sum(1):squeeze())
         input = torch.multinomial(probs:float(), 1):squeeze()
      else
         local m, am = o:max(1)
         input = am:squeeze()
      end
      table.insert(rv, input)
   end
   return rv
end