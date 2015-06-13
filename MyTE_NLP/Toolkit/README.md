# MyTE_Toolkit

MyTE_Toolkit is a library built to ease the development and use of natural language processing technologies. It includes modules for tokenization, ngram and matrix language modeling, frequency and probability distributions, and naive bayes classification. It also includes pythonic String, Set, DefaultTable, and Counter classes, and supplies many new utility functions to the table and torch libraries.

# Example Usage

## Tokenization

```lua
local st = nlp.String('words and words and word and ugh! 123. what is this?')
local lpt = nlp.LuaPatternsTokenizer('%p+')
print(lpt:tokenize(st))

local s = nlp.String('abcde')
print(nlp.bigrams(s))
print(nlp.trigrams(s))
print(nlp.ngrams(s, 2, true, true, 'yo'))
```

## Language Modeling

```lua
local text = 'Four score and seven years ago ...'
local tokens = nlp.word_tokenize(text)
local factory = function(fdist, bins) return nlp.LidstoneProbDist(fdist, 0.2) end
local lm = nlp.NgramLM(tokens, 3, factory)
print(lm, lm.model)

local context = 'four score'
print(lm(context))
print(lm:prob(context, 'and'))
print(lm:sample(context, 10))

local docs = {
   'the bird is the word',
   'how many dogs can eat the pizza',
   'why is this the way',
   'testing testing testing',
   'word and more words and more words'
}
local lsa = nlp.LSAModel(docs, true)
print(lsa:query('the bird is the word', true, true))
```

## Probability

```lua
require '../MyTE_NLP'

local C = nlp.Counter{a=5, b=4, c=3, d=6}
print(C.samples)

C = nlp.FreqDist{1, 1, 2, 2, 3, 3, 4}
print(C.samples)
print(C:freq(1), C:freq(4), C:freq(5))

C = nlp.FreqDist(nlp.String('hellllo'))
print(C.samples)

C = nlp.FreqDist(nlp.word_tokenize('hello world hello world'))
print(C.samples)

C = nlp.FreqDist(nlp.String('hellllo'):table())
print(C.samples)

print(C 'l')
print(C 'i')

C:push('l')
print(C.samples)
print(C)
print(C:freq('l'), C:freq('h'))

local upd = nlp.UniformProbDist{1, '2', 3, 4}
print(upd)
print(upd.samples)
print(upd:prob(1), upd(5))

local tpd = nlp.TableProbDist({a=0.4, b=0.5, c=0.05}, true, true)
print(tpd)
print(tpd.samples)
print(tpd:prob('a'), tpd 'e')

local mlepd = nlp.MLEProbDist(C)
print(mlepd)
print(mlepd:prob('l'))

local lpd = nlp.LidstoneProbDist(C, 0.2)
print(lpd)
print(lpd:prob('l'), lpd 'e', lpd 'g')

local sent = 'the the the dog dog some other words that we do not care about'
local tokens = nlp.word_tokenize(nlp.String(sent))
local cfdist = nlp.ConditionalFreqDist()
for n, word in pairs(tokens) do
   local condition = #word
   cfdist:push(condition, word)
end
print(cfdist)
print(cfdist(3)('the'))

local factory = function(fdist, bins) return nlp.LidstoneProbDist(fdist, 0.2) end
local cpdist = nlp.ConditionalProbDist(cfdist, factory, #cfdist)
print(cpdist)
print(cpdist(3):prob('the'))
print(cpdist(4):prob('that'))
```

## Naive Bayes

```lua
local data = {}
local ntrain = 100

for i = 1,  ntrain * 2 do
   local x = {
      math.random() + 5,
      math.random() + 5
   }
   local y = 1
   table.insert(data, {x, y})
end

for i = 1,  ntrain do
   local x = {
      math.random() - 5,
      math.random() - 5
   }
   local y = 0
   table.insert(data, {x, y})
end

torch.shuffle(data)
local model = nlp.NaiveBayesClassifier(data)
print(model)
print(model:classify{5, 5}:max())
```