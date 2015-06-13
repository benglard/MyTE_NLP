local Module, parent = torch.class('rl.Module', 'nn.Module')

function Module:__init(env, options)
   --[[
      REQUIRES:
         env -> a lua table with keys nstates and nactions,
            and values corresponding to the respective
            number of states and possible actions in the
            agent's environment
         options -> a lua table or nil
      EFFECTS:
         Creates an instance of rl.Module, a baseclass for use
         in building reinforcement learning architectures
   ]]

   parent.__init(self)
   self.env = env or {
      nstates = 2,
      nactions = 2
   }
   self.nstates  = self.env.nstates
   self.nactions = self.env.nactions
   self.options = options or {}

   self.nsteps = 0
   self.loss   = 0
   self.prev_r = nil
   self.prev_s = nil
   self.next_s = nil
   self.prev_a = nil
   self.next_a = nil

   self.w  = torch.Tensor()
   self.dw = torch.Tensor()
end

function Module:reset(stdv)
   self.nsteps = 0
   self.loss   = 0
   self.prev_r = nil
   self.prev_s = nil
   self.next_s = nil
   self.prev_a = nil
   self.next_a = nil
end

function Module:act(state)
   --[[
      REQUIRES:
         state -> number or Tensor
      EFFECTS:
         Perform an "action" on the state of the agent's
         environment. Here, throws a NotImplemented error
   ]]

   error('Module.act not implemented')
end

function Module:learn(reward)
   --[[
      REQUIRES:
         reward -> a number
      EFFECTS:
         Let the agent learn how to maximize it's
         reward. Here, throws a NotImplemented error
   ]]

   error('Module.learn not implemented')
end

function Module:updateOutput(input)
   -- Same as Module:act,
   -- for compatability with nn
   return self:act(input)
end

function Module:updateGradInput(reward)
   -- Same as Module:learn
   -- for compatability with nn
   return self:learn(reward)
end

function Module:update(net, method, params)
   --[[
      REQUIRES:
         net -> an instance of nn
         method -> an optimization method,
            either 'sgd' or 'rmsprop'
         params -> params of method
      EFFECTS:
         Updates the network using sgd
         or rmsprop
   ]]

   if method == 'sgd' then
      self:sgd(net, unpack(params))
   elseif method == 'rmsprop' then
      self:rmsprop(net, unpack(params))
   else
      error('unsupported optimization method')
   end
end

function Module:sgd(net, lr, momentum)
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
   mom = momentum or 0.9

   local ps, gs = net:getParameters()
   gs:mul(-lr)

   if mom > 0.0 then
      self.dw:resizeAs(gs):mul(mom):add(gs)
      ps:add(self.dw)
   else
      ps:add(gs)
   end

   gs:zero()
end

function Module:rmsprop(net, lr, clip)
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

   local ps, gs = net:getParameters()
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

function Module:push(state, action)
   --[[
      REQUIRES:
         state -> a number, index of current state
         action -> a number, index of current action
      EFFECTS:
         Shifts the agent's current state
   ]]

   assert(type(state) == 'number' and type(action) == 'number',
      'rl.Module.push requires input of type number')
   self.prev_s = self.next_s
   self.prev_a = self.next_a
   self.next_s = state
   self.next_a = action
end