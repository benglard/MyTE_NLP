package = 'MyTE_NLP'
version = 'scm-1'

source = {
   url = 'git://github.com/benglard/MyTE_NLP'
}

description = {
   summary = 'MyTE NLP Library',
}

dependencies = {
   'torch >= 7.0',
   'trepl',
   'nn',
   'nngraph',
   'optim',
   'xlua',
   'tds'
}

build = {
   type = 'command',
   install_command = 'cp -r MyTE_NLP $(LUADIR)'
}