local Tokenizer = torch.class('nlp.Tokenizer')

function Tokenizer:__init()
   --[[
      EFFECTS:
         Creates an instance of the nlp.Tokenizer
         class for use in tokenizing a string
   ]]
end

function Tokenizer:tokenize(s)
   --[[
      REQUIRES:
         s -> a string or nlp.String
      EFFECTS:
         Returns a lua table with values
         equivalent to the tokens of s
         split by space.
   ]]

   return s:split(' ')
end

nlp.word_tokenize = function(s)
   --[[
      REQUIRES:
         s -> a string or nlp.String
      EFFECTS:
         Creates an instance of the nlp.Tokenizer
         class and then tokenizes s
   ]]

   return nlp.Tokenizer():tokenize(s)
end