require 'trepl'
require '../MyTE_NLP'
require 'image'
print(rl)

local N = 40
local height = 480
local width = 640
local att = rl.ZoomableAttentionWindow(height, width, N)

local img = image.load('./data/catbw.jpg')
local x = img:clone():div(255)
local center_y = torch.Tensor{200.5}
local center_x = torch.Tensor{330.5}
local delta = torch.Tensor{5.0}
local sigma = torch.Tensor{2.0}

local W = att:read(x, center_y, center_x, delta, sigma)
local I2 = att:write(W, center_y, center_x, delta, sigma)

image.display(img:reshape(height, width))
image.display(W:reshape(N, N))
image.display(I2:reshape(height, width))