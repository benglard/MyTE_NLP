-------- BaseMatrixLM

local BaseMatrixLM, BaseLM = torch.class('nlp.BaseMatrixLM', 'nlp.BaseLM')

function BaseMatrixLM:__init(documents, transforms, preprocess, from, modelpath)
   --[[
      REQUIRES:
         documents -> a lua table
         transforms -> a set of matrix transformations to apply
            to the matrix language model, e.g. nlp.TFIDF
         preprocess -> if true, preprocesses the documents
         from -> if true, loads the model from a saved model
         modelpath -> it from is true, loads the model from the
            given modelpath
      EFFECTS:
         Creates an instance of the nlp.BaseMatrixLM class, a baseclass
         for use in building matrix language models.
   ]]

   documents = documents or {}
   self.transforms = transforms or {}
   if preprocess == nil then preprocess = false end
   self.preprocess = preprocess
   from = from or false
   self.model = model or ''
   if from then
      self:load(model)
   elseif documents then
      BaseLM.__init(self, documents)
   end
   if self.documents then
      self:make()
   end
end

function BaseMatrixLM:make()
   --[[
      EFFECTS:
         Creates a matrix representation of text where the
         rows -> documents / columns -> words and a row/col
         entry equals the term frequency of the word of that
         col in the doc of that row.
   ]]

   if self.preprocess then
      self.documents = nlp.preprocess(self.documents)
   end

   self.words = nlp.unique_words(self.documents)
   self.nwords = #self.words
   self.ndocs = #self.documents
   self.matrix = torch.zeros(self.ndocs, self.nwords)
   for d = 1, self.ndocs do
      self.matrix[d] = self:vectorize(self.documents[d])
   end

   -- Apply transforms
   for n, cls in pairs(self.transforms) do
      local t = cls(self.matrix)
      self.matrix = t:apply()
   end
end

function BaseMatrixLM:vectorize(doc)
   --[[
      REQUIRES:
         doc -> a string to vectorize
      EFFECTS:
         Makes a vector of length given by the number
         of unique words in all self.documents. For
         every word in this document, marks vector of 
         the index of that word in self.words equal to the
         term frequency.
   ]]

   local t = torch.type(doc)
   local tokens = {}
   local vector = torch.zeros(self.nwords)

   if t == 'string' or t == 'nlp.String' then
      tokens = doc:split(' ')
   elseif t == 'table' then
      tokens = doc
   else
      error('BaseMatrixLM.vectorize requires argument of type table or nlp.String')
   end

   for n, word in pairs(tokens) do
      local idx = table.index(self.words, word)
      if idx ~= -1 then
         local current = vector[idx]
         vector[idx] = current + 1
      end
   end
   return vector
end

function BaseMatrixLM:query(search, sort, preprocess)
   --[[
      REQUIRES:
         search -> a string to search by
         sort -> sorts if true else doesn't sort
         proceproces -> preprocesses if true else doesn't
      EFFECTS:
         Calculates the cosine similarity between
         the vector form of the search string, and
         every document in self.documents. If sort
         is true, returns a list of the documents
         sorted by similarity descending, otherwise
         returns a list of the similarity values.
   ]]

   sort = sort or false
   if preprocess == nil then preprocess = false end
   preprocess = preprocess
   local doc
   if preprocess then
      doc = nlp.preprocess(doc)
   else
      doc = search
   end

   local vector = self:vectorize(doc)
   local sims = {}
   for d = 1, self.ndocs do
      local row = self.matrix[d]
      sims[d] = {
         self.documents[d],
         torch.cossim(vector, row)
      }
   end

   if sort then
      table.sort(sims, function(a, b) return a[2] > b[2] end)
   end
   return sims
end

function load(path)
   --[[
      REQUIRES:
         path -> a path to load a BaseMatrixLM from
      EFFECTS:
         Loads a BaseMatrixLM from path
   ]]

   assert(type(path) == 'string',
      'BaseMatrixLM.load requires argument of type string')
   local data = torch.load(path)
   self.matrix    = data.matrix
   self.documents = data.documents
   self.words     = data.words
   self.nwords    = data.nwords
   self.ndocs     = data.ndocs
   return self
end

function save(path)
   --[[
      REQUIRES:
         path -> a path to save a BaseMatrixLM to
      EFFECTS:
         Saves a BaseMatrixLM to path
   ]]

   assert(type(path) == 'string',
      'BaseMatrixLM.save requires argument of type string')
   torch.save(path, {
      matrix    = self.matrix,
      documents = self.documents,
      words     = self.words,
      nwords    = self.nwords,
      ndocs     = self.ndocs
   })
end

-------- BaseMatrixTransform

local BaseMatrixTransform = torch.class('nlp.BaseMatrixTransform')

function BaseMatrixTransform:__init(m)
   --[[
      REQUIRES:
         m -> a torch Tensor to transform
      EFFECTS:
         Creates an instance of BaseMatrixTransform class
   ]]

   self.m = m
end

function BaseMatrixTransform:apply()
   --[[
      EFFECTS:
         Applies the matrix transformation. Here, it throws a
         NotImplementedError
   ]]

   error('BaseMatrixTransform.apply not implemented')
end

-------- TFIDF

local TFIDF, _ = torch.class('nlp.TFIDF', 'nlp.BaseMatrixTransform')

function TFIDF:apply()
   --[[
      EFFECTS:
         Applies TFIDF protocol to self.matrix        
         With a document-term matrix: matrix[x][y]
         tf[x][y] = frequency of term y in document x
         idf[x][y] = log( total number of documents in corpus / number of documents with term y )
   ]]

   self.N = self.m:size(2)
   self.sum = self.m:sum(1)
   local tfidf = self.m:clone()
   for col = 1, self.N do
      tfidf[{{}, col}]:mul(self:idf(col))
   end
   return tfidf
end

function TFIDF:idf(col)
   --[[
      REQUIRES:
         col -> the col of self.m to compute idf of
      EFFECTS:
         Computes log(N / # docs with value in col)
   ]]

   local c = self.sum[1][col]
   return math.log(self.N / c)
end

-------- LSA

local LSA, _ = torch.class('nlp.LSA', 'nlp.BaseMatrixTransform')

function LSA:apply()
   --[[
      EFFECTS:
         Applies SVD and then dimensionality reduction to a matrix 
         to uncover latent relationships.
         http://en.wikipedia.org/wiki/Latent_semantic_analysis
   ]]

   local U, s, V = torch.svd(self.m)
   local n = torch.random(1, 3)
   local l = s:size(1)
   s[{{l-n+1,l}}] = 0.0
   local S = torch.diagsvd(s, self.m:size(1), self.m:size(2))
   return U * S * V
end

-------- LexRank

local LexRank, _ = torch.class('nlp.LexRank', 'nlp.BaseMatrixTransform')

function LexRank:apply()
   --[[
      EFFECTS:
         Calculates the cosine similarity between every pair
         of sentences, remakes the matrix where
         matrix[i, j] = cos(matrix[i], matrix[j]). This
         matrix is symmetric.
   ]]

   local N = self.m:size(1)
   local rv = torch.zeros(N, N)
   for i = 1, N do
      for j = 1, N do
         local sim = torch.cossim(self.m[i], self.m[j])
         rv[i][j] = sim
         rv[j][i] = sim
      end
   end
   return rv
end

-------- Matrix Language Models

-- Perform TFIDF on documents

local TFIDFModel, _ = torch.class('nlp.TFIDFModel', 'nlp.BaseMatrixLM')

function TFIDFModel:__init(docs, ...)
   BaseMatrixLM.__init(self, docs, {nlp.TFIDF}, ...)
end

-- Perform TFIDF and LSA on documents

local LSAModel, _ = torch.class('nlp.LSAModel', 'nlp.BaseMatrixLM')

function LSAModel:__init(docs, ...)
   BaseMatrixLM.__init(self, docs, {nlp.TFIDF, nlp.LSA}, ...)
end

-- Perform TFIDF and LexRank on documents

local LexRankModel, _ = torch.class('nlp.LexRankModel', 'nlp.BaseMatrixLM')

function LexRankModel:__init(docs, ...)
   BaseMatrixLM.__init(self, docs, {nlp.TFIDF, nlp.LexRank}, ...)
end

-------- PageRank, ie. billions of USD in 13 LOC

nlp.pagerank = function(matrix, d, eps)
   --[[
      REQUIRES:
         matrix -> a torch Tensor of size 2
         d -> a number between 0 and 1 or nil
         eps -> a number near 0 or nil
      EFFECTS:
         Calculates the pagerank of matrix, ie.
         the principal eigenvector
   ]]

   d = d or 0.85
   eps = eps or 1e-4
   local N = matrix:shape(1)
   local v = torch.ones(N) / N
   local lastv = torch.ones(N) * 1e10 -- close to inf
   local M = matrix:clone():mul(d):add(torch.ones(N, N):mul((1 - d) / N))
   while torch.norm(torch.add(v, -lastv)) > eps do
      lastv = v
      v = M:mm(v)
   end
   return v
end