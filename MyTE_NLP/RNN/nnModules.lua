local Module = torch.getmetatable('nn.Module')
local Linear = torch.getmetatable('nn.Linear')

-------- nn.LinearNoBias

local LinearNoBias, _ = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
   Linear.__init(self, inputSize, outputSize)
   self.bias:zero()
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
   Linear.accGradParameters(self, input, gradOutput, scale)
   self.gradBias:zero()
end

-------- nn.ExpandAs

local ExpandAs, _ = torch.class('nn.ExpandAs', 'nn.Module')

function ExpandAs:__init()
   Module.__init(self)
end

function ExpandAs:updateOutput(input)
   local prev, other = unpack(input)
   self.output:resizeAs(prev):copy(prev):expandAs(self.output, other)
   return self.output
end

function ExpandAs:updateGradInput(input)
   local prev, gradOutput = unpack(input)
   self.gradInput
      :resizeAs(prev)
      :copy(prev)
      :expandAs(self.gradInput, gradOutput)
   return self.gradInput
end

-------- nn.NumberToTensor

local NumberToTensor, _ = torch.class('nn.NumberToTensor', 'nn.Module')

function NumberToTensor:__init()
   Module.__init(self)
end

function NumberToTensor:updateOutput(input)
   local t = type(input)
   if t == 'number' then
      return torch.Tensor(1):copy(input)
   elseif t == 'table' then
      local rv = {}
      for i = 1, #input do
         table.insert(rv, torch.Tensor(1):copy(input[i]))
      end
      return rv
   end
end

function NumberToTensor:updateGradInput(input, gradOutput)
   print(input, gradOutput)
end