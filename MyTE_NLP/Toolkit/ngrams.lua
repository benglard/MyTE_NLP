local _ngrams = function(sequence, n, pad_left, pad_right, pad_symbol)
   --[[
      REQUIRES:
         sequence -> a lua table or nlp.String
         n ->  a number, the n in ngram
         pad_left -> if true, pads the sequence at the start
         pad_right -> if true, pads the sequence at the end
         pad_symbol -> a string to pad if pad_left or pad_right are true
   ]]

   assert(sequence ~= nil and torch.type(sequence) == 'nlp.String' or torch.type(sequence) == 'table',
      'ngrams requires a first input of type table or nlp.String')
   assert(n ~= nil and type(n) == 'number',
      'ngrams requires a second input of type number')

   pad_left   = pad_left or false
   pad_right  = pad_left or false
   pad_symbol = pad_symbol or ''

   if type(sequence) == 'table' then
      local t = torch.type(sequence[1])
      if t == 'nlp.String' then
         for n, elem in pairs(sequence) do
            sequence[n] = elem.str
         end
      end
   elseif torch.type(sequence) == 'nlp.String' then
      sequence = sequence:table()
   end

   -- Pad
   if pad_left then
      for i = 1, n - 1 do
         table.push(sequence, pad_symbol)
      end
   end
   if pad_right then
      for i = 1, n - 1 do
         table.insert(sequence, pad_symbol)
      end
   end

   -- Start ngrams
   local history = {}
   for i = 1, n - 1 do
      table.insert(history, sequence[i])
   end

   -- Construct ngrams
   local rv = {}
   for i = n, #sequence do
      table.insert(history, sequence[i])
      table.insert(rv, table.clone(history))
      table.remove(history, 1)
   end

   return rv
end

return {
   ngrams = _ngrams,
   bigrams = function(s, pl, pr, ps)
      -- Compute bigrams
      return _ngrams(s, 2, pl, pr, ps)
   end,
   trigrams = function(s, pl, pr, ps)
      -- Compute trigrams
      return _ngrams(s, 3, pl, pr, ps)
   end
}