local _preprocess = function(docs)
   --[[
      REQUIRES:
         docs -> a lua table or string or nlp.String
      EFFECTS:
         Preprocesses the contents of docs by removing
         stop words and punctuation/numbers, etc.
   ]]

   if type(docs) ~= 'table' then
      docs = {docs}
   end

   local ow = nlp.OnlyWords()
   for n, elem in pairs(docs) do
      local tokens = ow:tokenize(elem)
      local nostop = {}
      for i, elem in pairs(tokens) do
         local s = elem.str
         if not nlp.ENGLISH_STOP_WORDS:contains(s) then
            table.insert(nostop, s)
         end
      end
      docs[n] = table.concat(nostop, ' ')
   end
   return docs
end

local _unique = function(docs)
   --[[
      REQUIRES:
         docs -> a lua table or string or nlp.String
      EFFECTS:
         Retuns a lua table storing the unique words
         in docs.
   ]]

   if type(docs) ~= 'table' then
      docs = {docs}
   end

   local words = nlp.Set()
   for n, elem in pairs(docs) do
      for i, word in pairs(elem:split(' ')) do
         words:add(word)
      end
   end
   return words:table()
end

return {
   preprocess = _preprocess,
   unique_words = _unique,
}