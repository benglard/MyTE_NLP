rnn.RecurrentAttention(
   rnn.FFAttention(n_input, batch_size, true):apply('att', opt.debug),
   rnn.Recurrent(n_input, n_hidden, batch_size, true):apply('rnn', opt.debug),
   seq_length
)