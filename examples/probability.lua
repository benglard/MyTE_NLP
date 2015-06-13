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
   --cfdist(condition):push(word, cfdist(condition)(word) + 1)
end
print(cfdist)
for n = 2, 5 do
   print(cfdist(n))
   print(cfdist(n).samples)
end
print(cfdist(3)('the'))

local factory = function(fdist, bins) return nlp.LidstoneProbDist(fdist, 0.2) end
local cpdist = nlp.ConditionalProbDist(cfdist, factory, #cfdist)
print(cpdist)
for n = 2, 5 do
   print(cpdist(n))
end
print(cpdist(3):prob('the'))
print(cpdist(4):prob('that'))