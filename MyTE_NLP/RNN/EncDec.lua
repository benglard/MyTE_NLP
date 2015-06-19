local EncDec, parent = torch.class('rnn.EncDec', 'rnn.Module')

function EncDec:__init(encoder, decoder)
   --[[
      REQUIRES:
         encoder -> an instance of nn.Module or nngraph.gModule
         decoder -> an instance of nn.Module or nngraph.gModule
      EFFECTS:
         Creates an instance of the rnn.EncDec class for use
         in building recurrent nueral network
         encoder-decoder models.
   ]]

   self.encoder = encoder
   self.decoder = decoder

   self.layer = nn.Sequential()
   self.layer:add(encoder)
   self.layer:add(decoder)
end

function EncDec:__tostring__()
   --[[
      EFFECTS:
         Returns the string representation of
         self.layer
   ]]

   return tostring(self.layer)
end