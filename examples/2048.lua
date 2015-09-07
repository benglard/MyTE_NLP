require '../MyTE_NLP'

local cmd = torch.CmdLine()
cmd:option('--hidden', 100, '# hidden units')
cmd:option('--agent', 'DeepQ', 'DeepQ | RecurrentDeepQ')
cmd:option('--size', 4, 'Game board size')
cmd:option('--layer', 'rec', 'rec | gru')
cmd:option('--nl', 1, '# layers')
cmd:option('--attend', false, 'use attention')
cmd:option('--seq', 4, 'seq length')
local opt = cmd:parse(arg or {})

table.reverse = function(t)
   local rv = {}
   local l = #t
   for i = l, 1, -1 do
      rv[l - i + 1] = t[i] 
   end
   return rv
end

-------- GameTile

local Tile = torch.class('GameTile')

function Tile:__init(position, value)
   self.x = position.x
   self.y = position.y
   self.value = value or 2
   self.prev_pos = nil
   self.merged_from = nil
end

function Tile:save()
   self.prev_pos = { x = self.x, y = self.y }
end

function Tile:update(position)
   self.x = position.x
   self.y = position.y
end

function Tile:serialize()
   return {
      position = {
         x = self.x,
         y = self.y
      },
      value = self.value
   }
end

-------- GameGrid

local Grid = torch.class('GameGrid')

function Grid:__init(size, prev_state)
   self.size = size
   if prev_state == nil then
      self.cells = self:empty()
   else
      self.cells = self:fromState(prev_state)
   end
end

function Grid:empty()
   local cells = {}
   for x = 1, self.size do
      cells[x] = {}
      for y = 1, self.size do
         cells[x][y] = nil
      end
   end
   return cells
end

function Grid:fromState(state)
   local cells = {}
   for x = 1, self.size do
      cells[x] = {}
      for y = 1, self.size do
         local tile = state[x][y]
         if tile == nil then
            cells[x][y] = nil
         else
            cells[x][y] = GameTile(tile.position, tile.vale)
         end
      end
   end
   return cells
end

function Grid:randomAvailableCell()
   local cells = self:availableCells()
   if #cells > 0 then
      local idx = torch.random(1, #cells)
      return cells[idx]
   end
end

function Grid:availableCells()
   local cells = {}
   self:eachCell(function(x, y, tile)
      if tile == nil then
         table.insert(cells, { x = x, y = y })
      end
   end)
   return cells
end

function Grid:eachCell(cb)
   for x = 1, self.size do
      for y = 1, self.size do
         cb(x, y, self.cells[x][y])
      end
   end
end

function Grid:cellsAvailable()
   return #self:availableCells() > 0
end

function Grid:cellAvailable(cell)
   return not self:cellOccupied(cell)
end

function Grid:cellOccupied(cell)
   return self:cellContent(cell) ~= nil
end

function Grid:cellContent(cell)
   if self:withinBounds(cell) then
      return self.cells[cell.x][cell.y]
   else
      return nil
   end
end

function Grid:insertTile(tile)
   self.cells[tile.x][tile.y] = tile
end

function Grid:removeTile(tile)
   self.cells[tile.x][tile.y] = nil
end

function Grid:withinBounds(position)
   return position.x >= 1 and position.x <= self.size and 
      position.y >= 1 and position.y <= self.size
end

function Grid:__tostring()
   local str = ''
   for x = 1, self.size do
      for y = 1, self.size do
         local tile = self.cells[x][y]
         if tile == nil then
            str = str .. 'e'
         else
            str = str .. tile.value
         end
         str = str  .. '\t'
      end
      str = str .. '\n'
   end
   return str
end

function Grid:serialize()
   local state = {}
   for x = 1, self.size do
      for y = 1, self.size do
         local cell = self.cells[x][y]
         if cell == nil then
            state[x][y] = nil
         else
            state[x][y] = cell:serialize()
         end
      end
   end
   return {
      size = self.size,
      cells = state
   }
end

-------- GameManager

local Manager = torch.class('GameManager')

function Manager:__init(size)
   self.size = size
   self.nstart = 2
   self.map = {
      [1] = { x = 0,  y = -1 }, -- Up
      [2] = { x = 1,  y = 0 },  -- Right
      [3] = { x = 0,  y = 1 },  -- Down
      [4] = { x = -1, y = 0 }   -- Left
   }
   self:setup()
end

function Manager:setup()
   self.over = false
   self.won = false
   self.grid = GameGrid(self.size)
   self.score = 0
   self:addStartTiles()
end

function Manager:addStartTiles()
   for i = 1, self.nstart do
      self:addRandomTile()
   end
end

function Manager:addRandomTile()
   if self.grid:cellsAvailable() then
      local value
      if math.random() < 0.9 then
         value = 2
      else
         value = 4
      end

      local tile = GameTile(self.grid:randomAvailableCell(), value)
      self.grid:insertTile(tile)
   end
end

function Manager:prepareTiles()
   self.grid:eachCell(function(x, y, tile)
      if tile ~= nil then
         tile.merged_from = nil
         tile:save()
      end
   end)
end

function Manager:moveTile(tile, cell)
   self.grid.cells[tile.x][tile.y] = nil
   self.grid.cells[cell.x][cell.y] = tile
   tile:update(cell)
end

function Manager:move(direction)
   -- 1: up, 2: right, 3: down, 4: left
   local cell, tile
   local vector = self:getVector(direction)
   local traversals = self:buildTraversals(vector)
   local moved = false
   self:prepareTiles()

   table.foreach(traversals.x, function(x)
      table.foreach(traversals.y, function(y)
         cell = { x = x, y = y }
         tile = self.grid:cellContent(cell)

         if tile ~= nil then
            local positions = self:findFarthestPosition(cell, vector)
            local next_c = self.grid:cellContent(positions.next_c)

            if next_c ~= nil
               and next_c.value == tile.value 
               and next_c.merged_from == nil then
               local merged = GameTile(positions.next_c, tile.value * 2)
               merged.merged_from = {tile, next}

               self.grid:insertTile(merged)
               self.grid:removeTile(tile)
               tile:update(positions.next_c)
               self.score = self.score + merged.value

               if merged.value == 2048 then
                  self.won = true
                  print('WOOHOO WE WON!!!')
                  print(self.grid)
                  os.exit()
               end
            else
               self:moveTile(tile, positions.farthest)
            end

            if not self:positionsEqual(cell, tile) then
               moved = true
            end
         end
      end)
   end)

   if moved then
      self:addRandomTile()
      if not self:movesAvailable() then
         self.over = true
         print('AWWWW WE LOST!!!')
         print(self.grid)
         self:setup() -- restart
      end
   end
end

function Manager:getVector(direction)
   assert(direction > 0 and direction < 5, 'bad direction')
   return self.map[direction]
end

function Manager:buildTraversals(vector)
   local traversals = { x = {}, y = {} }
   for pos = 1, self.size do
      table.insert(traversals.x, pos)
      table.insert(traversals.y, pos)
   end

   if vector.x == 1 then
      traversals.x = table.reverse(traversals.x)
   end
   if vector.y == 1 then
      traversals.y = table.reverse(traversals.y)
   end

   return traversals
end

function Manager:findFarthestPosition(cell, vector)
   local previous = cell
   cell = { x = previous.x + vector.x, y = previous.y + vector.y }

   while self.grid:withinBounds(cell) and self.grid:cellAvailable(cell) do
      previous = cell
      cell = { x = previous.x + vector.x, y = previous.y + vector.y }
   end

   return {
      farthest = previous,
      next_c = cell
   }
end

function Manager:movesAvailable()
   return self.grid:cellsAvailable() or self:tileMatchesAvailable()
end

function Manager:tileMatchesAvailable()
   local tile
   for x = 1, self.size do
      for y = 1, self.size do
         tile = self.grid:cellContent{x=x, y=y}
         if tile ~= nil then
            for direction = 1, 4 do
               local vector = self:getVector(direction)
               local cell = { x = x + vector.x, y = y + vector.y }
               local other = self.grid:cellContent(cell)
               if other ~= nil and other.value == tile.value then
                  return true
               end
            end
         end
      end
   end
   return false
end

function Manager:positionsEqual(first, second)
   return first.x == second.x and first.y == second.y
end

function Manager:state()
   local vec = torch.zeros(self.size, self.size)
   self.grid:eachCell(function(x, y, tile)
      if tile ~= nil then
         vec[x][y] = tile.value
      end
   end)
   return vec:resize(vec:nElement())
end

-------- Train!

local Game = GameManager(opt.size)
local sm = nn.SoftMax()

local env = { nstates = opt.size * opt.size, nactions = 4 }
local aopt = { rectifier = nn.ReLU, optim = 'rmsprop', memory = 1000000,
   interval = 1, gamma = 0.99, batchsize = 20, gradclip = 5, usestate = true,
   hidden = opt.hidden, rnntype = opt.layer, nlayers = opt.nl,
   attend = opt.attend, seq = opt.seq }
local params = { verbose = true }

local lastD = -1

local agent = rl[opt.agent](env, aopt)
print(opt, env, aopt, agent)

rl.RLTrainer(agent,
   function()
      return Game:state()
   end,
   function(action)
      local ps = sm:forward(action)
      local max, argmax = ps:max(1)
      local direction = argmax:squeeze()
      if direction == lastD then
         ps[direction] = 0.0
         max, argmax = ps:max(1)
         direction = argmax:squeeze()
      end
      --print(ps, direction)
      lastD = direction
      --local direction = torch.multinomial(ps, 1):squeeze()
      Game:move(direction)
      --print(Game.grid)
      return Game.score
   end
):train(params)