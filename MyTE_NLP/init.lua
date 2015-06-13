require 'torch'
require 'trepl'
require 'nn'
require 'nngraph'
require 'optim'
require 'luaimport'
require 'xlua'

require './RNN'
require './RL'
require './Toolkit'

math.randomseed(os.time())
torch.manualSeed(os.time())