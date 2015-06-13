local RecurrentReinforce, parent = torch.class('rl.RecurrentReinforce', 'rl.Reinforce')

function RecurrentReinforce:__init(env, options, rnnoptions)
   --[[
      REQUIRES:
         env -> a lua table with keys nstates and nactions,
            and values corresponding to the respective
            number of states and possible actions in the
            agent's environment
         options -> a lua table or nil
         rnnoptions -> a lua table specifying the type of rnn
            (rnn, lstm, gru) and the arguments to build the
            rnn layer.
      EFFECTS:
         Creates an instance of rl.RecurrentReinforce, a reinforcement learning
         agent that approximates the value function using a
         recurrent neural network to maximize future reward.
   ]]

   self.rnn = (options or {}).rnn or 'rnn'
   self.rnnoptions = rnnoptions or { args = {} }
   self.debug = self.rnnoptions.debug or false
   self.name  = self.rnnoptions.name  or ''
   parent.__init(self, env, options)
end

function RecurrentReinforce:_build()
   --[[
      EFFECTS:
         Returns the underlying network used to
         train the Reinforce agent.
   ]]

   local args = self.rnnoptions.args
   local model = nn.Sequential()
   local layer

   if self.rnn == 'rnn' then
      layer = rnn.Recurrent(unpack(args))
   elseif self.rnn == 'lstm' then
      model:add(nn.Linear(self.nstates, self.hidden))
      layer = rnn.LSTM(unpack(args))
   elseif self.rnn == 'gru' then
      layer = rnn.GRU(unpack(args))
   else
      error('Invalid recurrence type')
   end

   layer:apply(self.name, self.debug)
   model:add(layer)
   model:add(nn.Linear(layer.hiddenSize, self.nactions))
   return model
end