local Draw, parent = torch.class('rnn.Draw', 'rnn.Module')

function Draw:__init(input, hidden, batch, seq, output, width, height, window, verbose)
   --[[
      REQUIRES:
         input -> a number
         hidden -> a number
         batch -> a number
         seq -> a number
         output -> a number
         width -> a number
         height -> a number
         window -> a number
         verbose -> a boolean
      EFFECTS:
         Creates an instance of the rnn.Draw class for use
         in building Draw-like recurrent attention models.
         http://jmlr.org/proceedings/papers/v37/gregor15.pdf
   ]]

   parent.__init(self, input, hidden, batch, seq)
   self.outputSize = output
   self.A = width or math.sqrt(input)
   self.B = height or self.A
   self.N = window
   self.tol = 1e-4
   self.verbose = verbose or false
end

function Draw:build(annotate)
   self.doannotate = annotate or false

   self:log 'Making encoder'
   local encoder = self:buildEncoder(annotate)
   self:log 'Making encoder clones'
   self.encoder = nn.Sequential():add(encoder)
      :clone(self.seqSize, self.verbose)

   self:log 'Making decoder'
   local decoder = self:buildDecoder(annotate)
   self:log 'Making decoder clones'
   self.decoder = nn.Sequential():add(decoder)
      :clone(self.seqSize, self.verbose)

   self.layer = nn.Sequential()
   self.layer:add(self.encoder)
   self.layer:add(self.decoder)
   self:restart()
   return self
end

function Draw:ncopies(x, dim)
   dim = dim or self.A
   local y = nn.Reshape(1)(x)
   local t = {}
   for i = 1, dim do table.insert(t, nn.Copy()(y))
   end
   return nn.JoinTable(2)(t)
end

function Draw:filterbank(gamma, dim, patch, delta, sigma)
   local filters = {}
   for i = 1, self.N do
      local mu = nn.MulConstant(-1)(nn.CAddTable(){
         gamma, nn.MulConstant(i - self.N/2 - 1/2)(delta)
      })
      -- numerator
      local F = nn.Exp()(nn.CMulTable(){
         nn.Power(2)(nn.CAddTable(){ mu, patch }), sigma
      })
      -- normalizing factor
      --[[local Z = nn.ExpandAs()
         { nn.View(1, dim)(nn.AddConstant(self.tol)(nn.Sum()(F))),  F }
      local filter = nn.View(self.batchSize, 1, dim)(nn.CDivTable(){ F, Z })]]
      local filter = nn.View(self.batchSize, 1, dim)(F)
      table.insert(filters, filter)
      if self.doannotate then nngraph.annotateNodes() end
   end
   return nn.JoinTable(2)(filters)
end

function Draw:LSTM(x, prev_c, prev_h, input)
   local i2h = nn.Linear(input, 4 * self.hiddenSize)(x)
   local h2h = nn.Linear(self.hiddenSize, 4 * self.hiddenSize)(prev_h)
   local gates = nn.CAddTable(){ i2h, h2h }
   local reshaped = nn.Reshape(4, self.hiddenSize)(gates)
   local sliced = nn.SplitTable(2)(reshaped)
   local in_gate = nn.Sigmoid()(nn.SelectTable(1)(sliced))
   local in_transform = nn.Tanh()(nn.SelectTable(2)(sliced))
   local forget_gate = nn.Sigmoid()(nn.SelectTable(3)(sliced))
   local out_gate = nn.Sigmoid()(nn.SelectTable(4)(sliced))
   local memory = nn.CMulTable(){ forget_gate, prev_c }
   local write = nn.CMulTable(){ in_gate, in_transform }
   local next_c = nn.CAddTable(){ memory, write }
   local next_h = nn.CMulTable(){ out_gate, nn.Tanh()(next_c) }
   if self.doannotate then nngraph.annotateNodes() end
   return next_c, next_h
end

function Draw:buildEncoder()
   local hidden = self.hiddenSize
   local output = self.outputSize

   local x = nn.Identity()()
   local prev_err_x = nn.Identity()()
   local prev_dec_h = nn.Identity()()
   local patch_dim = nn.Identity()()
   local prev_c = nn.Identity()()
   local prev_h = nn.Identity()()
   local e = nn.Identity()()

   local gx = nn.MulConstant((self.A + 1) / 2)(nn.AddConstant(1)(
      self:ncopies(nn.Linear(hidden, 1)(prev_dec_h), self.A)))
   local gy = nn.MulConstant((self.B + 1) / 2)(nn.AddConstant(1)(
      self:ncopies(nn.Linear(hidden, 1)(prev_dec_h), self.B)))

   local delta = nn.Exp()(self:ncopies(nn.Linear(hidden, 1)(prev_dec_h)))
   local gamma = nn.Exp()(self:ncopies(nn.Linear(hidden, 1)(prev_dec_h)))
   local sigma = nn.Exp()(self:ncopies(nn.Linear(hidden, 1)(prev_dec_h)))
   local factor = (math.max(self.A, self.B) - 1) / (self.N - 1)
   delta = nn.MulConstant(factor)(delta)
   sigma = nn.MulConstant(-1/2)(nn.Power(-2)(sigma))

   local fx = self:filterbank(gx, self.A, patch_dim, delta, sigma)
   local fy = self:filterbank(gy, self.B, patch_dim, delta, sigma)

   local patch = nn.MM(false, true){ nn.MM(){ fx, x }, fy }
   local patch_error = nn.MM(false, true){ nn.MM(){ fx, prev_err_x }, fy }
   local n_in = 2 * self.N * self.N
   local read_in = nn.Reshape(n_in)(nn.JoinTable(3){ patch, patch_error })
   
   local next_c, next_h = self:LSTM(read_in, prev_c, prev_h, n_in)
   local mu = nn.Linear(hidden, output)(next_h)
   local sigma = nn.Exp()(nn.Linear(hidden, output)(next_h))

   local z = nn.CMulTable(){ mu, nn.CMulTable(){ sigma, e } }
   local mu_squared = nn.Square()(mu)
   local sigma_squared = nn.Square()(sigma)
   local log_sigma_sq = nn.Log()(sigma_squared)
   local minus_log_sigma = nn.MulConstant(-1)(log_sigma_sq)
   local loss_z = nn.Sum(2)(nn.MulConstant(0.5)(nn.AddConstant(-1)(
      nn.CAddTable(){ mu_squared, sigma_squared, minus_log_sigma }
   )))

   if self.doannotate then nngraph.annotateNodes() end
   return nn.gModule(
      {x, prev_err_x, prev_dec_h, patch_dim, prev_c, prev_h, e},
      {patch, next_c, next_h, z, loss_z})
end

function Draw:buildDecoder()
   local hidden = self.hiddenSize
   local output = self.outputSize

   local x = nn.Identity()()
   local z = nn.Identity()()
   local prev_c = nn.Identity()()
   local prev_h = nn.Identity()()
   local prev_canvas = nn.Identity()()
   local patch_dim = nn.Identity()()
   local n_in = output

   local next_c, next_h = self:LSTM(z, prev_c, prev_h, n_in)
   local gx = nn.MulConstant((self.A + 1) / 2)(nn.AddConstant(1)(
      self:ncopies(nn.Linear(hidden, 1)(next_h))))
   local gy = nn.MulConstant((self.B + 1) / 2)(nn.AddConstant(1)(
      self:ncopies(nn.Linear(hidden, 1)(next_h))))

   local delta = nn.Exp()(self:ncopies(nn.Linear(hidden, 1)(next_h)))
   local gamma = nn.Exp()(self:ncopies(nn.Linear(hidden, 1)(next_h)))
   local sigma = nn.Exp()(self:ncopies(nn.Linear(hidden, 1)(next_h)))
   local factor = (math.max(self.A, self.B) - 1) / (self.N - 1)
   delta = nn.MulConstant(factor)(delta)
   sigma = nn.MulConstant(-1/2)(nn.Power(-2)(sigma))
   
   local fx = self:filterbank(gx, self.A, patch_dim, delta, sigma)
   local fy = self:filterbank(gy, self.B, patch_dim, delta, sigma)

   local next_w = nn.Reshape(self.N, self.N)(
      nn.Linear(hidden, self.N * self.N)(next_h))
   local write = nn.MM(){ nn.MM(true, false){ fy, next_w }, fx }
   local next_canvas = nn.CAddTable(){ prev_canvas, write }
   local mu = nn.Sigmoid()(next_canvas)
   local diff = nn.CAddTable(){ x, nn.MulConstant(-1)(mu) }
   local loss_x = nn.Sum(2)(nn.Sum(3)(nn.Power(2)(diff)))
   local x_pred = nn.Reshape(self.A, self.B)(mu)
   local x_error = nn.Reshape(self.A, self.B)(diff)

   if self.doannotate then nngraph.annotateNodes() end
   return nn.gModule(
      {x, z, prev_c, prev_h, prev_canvas, patch_dim},
      {x_pred, x_error, next_c, next_h, next_canvas, loss_x})
end

function Draw:updateOutput(input)
   local input, patch_dim = unpack(input)

   local t = self.step
   local enc = self.encoder.clones[t]
   local dec = self.decoder.clones[t]
   local inputs, outputs

   self.e[t] = torch.randn(self.batchSize, self.outputSize)
   self.x[t] = input

   inputs = {
      input, self.x_error[t-1], self.dec_prev_h[t-1], patch_dim, 
      self.enc_prev_c[t-1], self.enc_prev_h[t-1], self.e[t]
   }
   outputs = enc:updateOutput(inputs)
   local patch, next_c, next_h, z, loss_z = unpack(outputs)
   self.patch[t] = patch
   self.enc_prev_c[t] = next_c
   self.enc_prev_h[t] = next_h
   self.z[t] = z
   self.loss.z[t] = loss_z

   inputs = {
      input, z, self.dec_prev_c[t-1], self.dec_prev_h[t-1],
      self.canvas[t-1], patch_dim
   }
   outputs = dec:updateOutput(inputs)
   local x_pred, x_error, next_c, next_h, next_canvas, loss_x = unpack(outputs)
   self.x_pred[t] = x_pred
   self.x_error[t] = x_error
   self.dec_prev_c[t] = next_c
   self.dec_prev_h[t] = next_h
   self.canvas[t] = next_canvas
   self.loss.x[t] = loss_x

   self.loss_t = self.loss_t + self.loss.z[t]:mean() + self.loss.x[t]:mean()

   if self.step ~= self.seqSize then
      self.step = t + 1
   end
end

function Draw:backward(patch_dim)
   local t = self.step
   local enc = self.encoder.clones[t]
   local dec = self.decoder.clones[t]
   local inputs, dinputs, outputs

   self.dloss.x[t] = torch.ones(self.batchSize, 1)
   self.dloss.z[t] = torch.ones(self.batchSize, 1)
   self.dx_pred[t] = torch.zeros(self.batchSize, self.A, self.B)
   self.dpatch[t] = torch.zeros(self.batchSize, self.N, self.N)

   inputs = {
      self.x[t], self.z[t], self.dec_prev_c[t-1], self.dec_prev_h[t-1],
      self.canvas[t-1], patch_dim
   }
   dinputs = {
      self.dx_pred[t], self.dx_error[t],
      self.dec_dprev_c[t], self.dec_dprev_h[t],
      self.dcanvas[t], self.dloss.x[t]
   }
   outputs = dec:backward(inputs, dinputs)
   local dx1, dz, next_c, next_h, next_canvas, _ = unpack(outputs)
   self.dx1[t] = dx1
   self.dz[t] = dz
   self.dec_dprev_c[t - 1] = next_c
   self.dec_dprev_h1[t - 1] = next_h
   self.dcanvas[t - 1] = next_canvas

   inputs = {
      self.x[t], self.x_error[t-1], self.dec_prev_h[t-1], patch_dim, 
      self.enc_prev_c[t-1], self.enc_prev_h[t-1], self.e[t]
   }
   dinputs = {
      self.dpatch[t], self.enc_dprev_c[t], self.enc_dprev_h[t],
      self.dz[t], self.dloss.z[t]
   }
   outputs = enc:backward(inputs, dinputs)
   local dx2, dx_err, dec_next_h, _, next_c, next_h, de = unpack(outputs)
   self.dx2[t] = dx2
   self.dx_error[t - 1] = dx_err
   self.dec_dprev_h2[t - 1] = dec_next_h
   self.enc_dprev_c[t - 1] = next_c
   self.enc_dprev_h[t - 1] = next_h
   self.de[t] = de

   self.dec_dprev_h[t-1] = self.dec_dprev_h1[t-1] + self.dec_dprev_h2[t-1]

   if self.step ~= 1 then
      self.step = t - 1
   end
end

function Draw:generate(input, patch_dim, save, path)
   save = save or false
   path = path or paths.cwd()
   local inputs, outputs

   for t = 1, self.seqSize do
      self.x[t] = input
      self.e[t] = torch.randn(self.batchSize, self.outputSize)
      self.z[t] = torch.randn(self.batchSize, self.outputSize)

      inputs = {
         input, self.z[t], self.dec_prev_c[t-1], self.dec_prev_h[t-1],
         self.canvas[t-1], patch_dim
      }
      outputs = self.decoder.clones[t]:updateOutput(inputs)
      local x_pred, x_error, next_c, next_h, next_canvas, loss_x = unpack(outputs)
      self.x_pred[t] = x_pred
      self.x_error[t] = x_error
      self.dec_prev_c[t] = next_c
      self.dec_prev_h[t] = next_h
      self.canvas[t] = next_canvas
      self.loss.x[t] = loss_x
   end

   if save then
      for t = 1, self.seqSize do
         local fname = string.format('%s/pred_%d.jpg', path, t)
         image.save(fname, self.x_pred[t][1])
         print('Saved to: ' .. fname)
      end
   end

   self:restart()
end

function Draw:batchXhidden(backward)
   local idx = 0
   if backward then idx = self.seqSize end
   return {[idx] = torch.zeros(self.batchSize, self.hiddenSize)}
end

function Draw:batchXdims(backward)
   local idx = 0
   if backward then idx = self.seqSize end
   return {[idx] = torch.zeros(self.batchSize, self.A, self.B)}
end

function Draw:restart()
   self.inputs = { enc = {}, dec = {} }
   self.outputs = { enc = {}, dec = {} }

   self.enc_prev_c = self:batchXhidden()
   self.enc_prev_h = self:batchXhidden()
   self.dec_prev_c = self:batchXhidden()
   self.dec_prev_h = self:batchXhidden()

   self.enc_dprev_c = self:batchXhidden(true)
   self.enc_dprev_h = self:batchXhidden(true)
   self.dec_dprev_c = self:batchXhidden(true)
   self.dec_dprev_h = self:batchXhidden(true)
   self.dec_dprev_h1 = self:batchXhidden(true)
   self.dec_dprev_h2 = self:batchXhidden(true)

   self.x = {}
   self.dx1 = {}
   self.dx2 = {}

   self.e = {}
   self.de = {}

   self.z = {}
   self.dz = {}

   self.x_pred = {}
   self.dx_pred = {}

   self.x_error = self:batchXdims()
   self.dx_error = self:batchXdims(true)

   self.patch = {}
   self.dpatch = {}

   self.canvas = self:batchXdims()
   self.dcanvas = self:batchXdims(true)

   self.loss = { x = {}, z = {} }
   self.dloss = { x = {}, z = {} }
   self.loss_t = 0
end

function Draw:apply(enc, dec, debug)
   self:name(enc or '', dec or '')
   if debug ~= nil then
      self:debug(debug)
   end
   return self
end

function Draw:name(enc, dec)
   for i, clone in ipairs(self.encoder.clones) do
      local gmod = clone.modules[1]
      gmod.name = enc .. i
      if self.verbose then gmod.verbose = true end
   end
   for i, clone in ipairs(self.decoder.clones) do
      local gmod = clone.modules[1]
      gmod.name = dec .. i
      if self.verbose then gmod.verbose = true end
   end
   return self
end

function Draw:log(str)
   if self.verbose then print(str) end
end

function Draw:__tostring__()
   local template = 'rnn.Draw(%d -> %d -> (%d, %d) | N=%d)'
   return string.format(template, self.inputSize, self.hiddenSize,
      self.A, self.B, self.N)
end