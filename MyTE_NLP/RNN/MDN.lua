local MDN, parent = torch.class('rnn.MDN', 'rnn.Module')

function MDN:__init(input, hidden, output, annotate)
   parent.__init(self, input, hidden)
   self.nlabels = output

   local x = nn.Identity()()
   local y = nn.Identity()()

   local mu = nn.Reshape(output, hidden)(
      nn.Linear(input, output * hidden)(x))
   local sigma = nn.Exp()(nn.Linear(input, hidden)(x))
   local alpha = nn.SoftMax()(nn.Linear(input, hidden)(x))

   local t = {}
   for i = 1, hidden do
      local z = nn.MulConstant(-1)(nn.Select(3, i)(mu))
      local s = nn.MulConstant(-0.5)(
         nn.Sum(2)(nn.Square()(nn.CAddTable(){ z, y })))
      local sigma_2 = nn.Select(2, i)(sigma)
      local sigma_sq_inv = nn.Power(-2)(sigma_2)
      local alpha_2 = nn.Select(2, i)(alpha)
      local e = nn.Exp()(nn.CMulTable(){ s, sigma_sq_inv })
      local sigma_mm = nn.Power(-output)(sigma_2)
      local factor = math.pow((2 * math.pi), -0.5 * output)
      local rv = nn.MulConstant(factor)(
         nn.CMulTable(){ e, sigma_mm, alpha_2 })
      table.insert(t, rv)
   end

   local res = nn.MulConstant(-1)(nn.Log()(nn.CAddTable()(t)))
   if annotate then nngraph.annotateNodes() end
   self.layer = nn.gModule({x, y}, {res, alpha, mu, sigma})
end

function MDN:updateOutput(input)
   self.input = input
   self.outputs = self.layer:forward(input)
   local output, alpha, mu, sigma = unpack(self.outputs)
   return output
end

function MDN:backward()
   local output, alpha, mu, sigma = unpack(self.outputs)
   local doutput = torch.ones(output:size())
   local dalpha = torch.zeros(alpha:size())
   local dmu = torch.zeros(mu:size())
   local dsigma = torch.zeros(sigma:size())
   self.dinputs = {doutput, dalpha, dmu, dsigma}
   return self.layer:backward(self.inputs, self.dinputs)
end