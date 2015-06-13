require '../MyTE_NLP'
print(nlp)

local s = nlp.String('hello world')
print(s, #s)

local e = s(2)
print(e)

local e = s{2}
print(e)

local sub = s{2, 4}
print(sub)

local subskip = s{2, 8, 2}
print(subskip)

local get = s:get{8, 2, -1}
print(get)

local reverse = s:reverse()
print(reverse)

local up = s:run(string.upper)
print(up)

local up = s:upper()
print(up)

local s2 = nlp.String('%s %s %d')
local f = s2:run(string.format, 'hello', 'world!', '42')
print(f)

local s2 = nlp.String('%s %s %d')
local f = s2:format('hello', 'world!', '42')
print(f, #f, f:__len(), f:len())

local s3 = s + nlp.String('\t') + f + ' yo'
print(s3)

local s4 = s3 * 3
print(s4)

local t = nlp.String('hello')
print(t:table())

for elem in t:iter() do
   print(elem)
end

local st = nlp.String('hello world hello world')
local to = nlp.Tokenizer()
print(to:tokenize(st))

local st = nlp.String('words and words and word and ugh! 123. what is this?')
local lpt = nlp.LuaPatternsTokenizer('%p+')
print(lpt:tokenize(st))
lpt.pattern = '%d+'
print(lpt:tokenize(st))

local ow = nlp.OnlyWords('%a+')
print(ow:tokenize(st))

local crazy = nlp.String('THE (QUICK) brOWN FOx JUMPS')
lpt.pattern = '%f[%a]%u+%f[%A]'
print(lpt:tokenize(crazy))

local s = nlp.String('abcde')
print(nlp.bigrams(s))
print(nlp.trigrams(s))
print(nlp.ngrams(s, 2, true, true, 'yo'))

print(nlp.word_tokenize('a b c d e'))