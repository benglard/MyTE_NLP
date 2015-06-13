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
      end
   end
end