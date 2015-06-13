local LuaPatternsTokenizer, LPTParent = torch.class('nlp.LuaPatternsTokenizer', 'nlp.Tokenizer')

function LuaPatternsTokenizer:__init(pattern)
   --[[
      EFFECTS:
         Creates an instance of the nlp.LuaPatternsTokenizer
         class for use in tokenizing a string using
         lua patterns
   ]]

   self.pattern = pattern
end

function LuaPatternsTokenizer:tokenize(s)
   --[[
      REQUIRES:
         s -> a string or nlp.String
      EFFECTS:
         Returns a lua table with values
         equivalent to the tokens of s
         split by the lua pattern.
   ]]

   local rv = {}
   for word in s:gmatch(self.pattern) do
      table.insert(rv, nlp.String(word))
   end
   return rv
end

local OnlyWords, _ = torch.class('nlp.OnlyWords', 'nlp.LuaPatternsTokenizer')

function OnlyWords:__init()
   --[[
      EFFECTS:
         Creates an instance of the nlp.OnlyWords
         class for use in tokenizing a string to
         only return words
   ]]

   self.pattern = '%a+'
end