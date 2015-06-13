local ZoomableAttentionWindow = torch.class('rl.ZoomableAttentionWindow')

function ZoomableAttentionWindow:__init(height, width, N)
   --[[
      REQUIRES:
         height -> height of an image
         width -> width of an image
         N -> size of window
      EFFECTS:
         Creates an instance of rl.ZoomableAttentionWindow
         for use in building neural network attention
         mechanisms. Note that while soft attention mechanisms do not
         use reinforcement learning, this module fits best in
         MyTE_RL.
   ]]

   self.img_height = height
   self.img_width = width
   self.N = N
end

function ZoomableAttentionWindow:filterbank(center_y, center_x, delta, sigma)
   --[[
      REQUIRES:
         center_y -> y coord of center
         center_x -> x coord of center
         delta -> stride of attention mechanism
         sigma -> std. dev of attention mechanism
      EFFECTS:
         Creates filterbank matrices of attention mechanism,
         FY and FX, given by equations (25) and (26) here:
         http://arxiv.org/pdf/1502.04623.pdf
   ]]

   local tol = 1e-4
   local N = self.N
   local batch_size = center_x:size(1)
   local range = torch.range(0, N - 1):add(-N/2):add(-0.5)

   local muX = torch.badd(center_x, torch.bmul(delta, range)):adddim()
   local muY = torch.badd(center_y, torch.bmul(delta, range)):adddim()

   local a = torch.range(0, self.img_width - 1)
   local b = torch.range(0, self.img_height - 1)

   local s  = sigma:pow(2)
   local FX = torch.badd(a, -muX):pow(2):div(-2):bdiv(s):exp()
   local FY = torch.badd(a, -muY):pow(2):div(-2):bdiv(s):exp()
   FX:cdiv(FX:sum(1):add(tol))
   FY:cdiv(FY:sum(1):add(tol))
   
   return FY, FX
end

function ZoomableAttentionWindow:read(images, center_y, center_x, delta, sigma)
   --[[
      REQUIRES:
         images -> a batch of torch tensors
         center_y -> a batch of center y coords
         center_x -> a batch of center x coords
         delta -> a batch of ZAW strides
         sigma -> a batch of ZAW std. dev's
      EFFECTS:
         Extracts a batch of attention windows
   ]]

   local N = self.N
   local batch_size = images:size(1)

   -- Reshape input into proper 2d images
   local I = images:reshape(batch_size, self.img_width, self.img_height)

   -- Get separable filterbank
   local FY, FX = self:filterbank(center_y, center_x, delta, sigma)

   -- apply to the batch of images
   local b1 = torch.bmm(FY, I):transpose(2, 3)
   local b2 = torch.bmm(b1, FX)
   local W = b2:resize(batch_size, N * N)
   return W
end

function ZoomableAttentionWindow:write(windows, center_y, center_x, delta, sigma)
   --[[
      REQUIRES:
         windows -> a batch of attention windows
         center_y -> a batch of center y coords
         center_x -> a batch of center x coords
         delta -> a batch of ZAW strides
         sigma -> a batch of ZAW std. dev's
      EFFECTS:
         Writes a batch of windows into full sized images.
   ]]

   local N = self.N
   local batch_size = windows:size(1)

   -- Reshape input into proper 2d windows
   local W = windows:reshape(batch_size, N, N)

   -- Get separable filterbank
   local FY, FX = self:filterbank(center_y, center_x, delta, sigma)

   -- apply...
   local b1 = torch.bmm(FY:transpose(2,3), W)
   local b2 = torch.bmm(b1, FX)
   local I = b2:resize(batch_size, self.img_height * self.img_width)
   return I
end