def part1():
    # data I/O
    data = open('shakespeare_train.txt', 'r').read() # should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print 'data has %d characters, %d unique.' % (data_size, vocab_size)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    # hyperparameters
    hidden_size = 100 # size of hidden layer of neurons
    seq_length = 25 # number of steps to unroll the RNN for
    learning_rate = 1e-1
    
    
    a = [0.01, 0.03, 0.05,0.07,0.09, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 3, 5, 7, 9, 10]
    
    n, p = 0, 0
    
    for alpha in a:
        
        print '--------\n', 'Temperature ==', 1/float(alpha), '\n--------' 
        
        # hyperparameters
        hidden_size = 100 # size of hidden layer of neurons
        seq_length = 25 # number of steps to unroll the RNN for
        learning_rate = 1e-1
        
        if p+seq_length+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
        
        # sample from the model now and then
        if n % 100 == 0:
        sample_ix = sample_w_temp(hprev, inputs[0], 200, 1)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print '----\n %s \n----' % (txt, )
        
        p+= seq_length # move data pointer
        n += 1 # iteration counter 