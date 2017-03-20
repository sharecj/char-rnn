#coding=utf8
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1

with open('anna.txt', 'r') as f:
    text=f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
chars = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

def split_data(chars, batch_size, num_steps, split_frac=0.9):
    """ 
    Split character data into training and validation sets, inputs and targets for each set.
    
    Arguments
    ---------
    chars: character array
    batch_size: Size of examples in each of batch
    num_steps: Number of sequence steps to keep in the input and pass to the network
    split_frac: Fraction of batches to keep in the training set
    
    
    Returns train_x, train_y, val_x, val_y
    """
    
    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)
    
    # Drop the last few characters to make only full batches
    x = chars[: n_batches*slice_size]
    y = chars[1: n_batches*slice_size + 1]
    
    # Split the data into batch_size slices, then stack them into a 2D matrix 
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))
    # np.split是指把x等分成几份
    
    # Now x and y are arrays with dimensions batch_size x n_batches*num_steps
    
    # Split into training and validation sets, keep the first split_frac batches for training
    split_idx = int(n_batches*split_frac)
    train_x, train_y= x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
    
    print('slice_size:%s' % slice_size)
    print('n_batches:%s' % n_batches)
    
    return train_x, train_y, val_x, val_y
    
def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape
    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]
                 
def build_rnn(num_classes, batch_size=50, num_steps=50, lstm_size=128, num_layers=2,
              learning_rate=0.001, grad_clip=5, sampling=False):
    
    # When we're using this network for sampling later, we'll be passing in
    # one character at a time, so providing an option for that
    if sampling == True:
        batch_size, num_steps = 1, 1

    tf.reset_default_graph()
    
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    
    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # One-hot encoding the input and target characters
    x_one_hot = tf.one_hot(inputs, num_classes)
    y_one_hot = tf.one_hot(targets, num_classes)

    ### Build the RNN layers
    # Use a basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    ### Run the data through the RNN layers
    # This makes a list where each element is on step in the sequence
    rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot, num_steps, 1)]
    
    # Run each sequence step through the RNN and collect the outputs
    outputs, state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=initial_state)
    final_state = state
    
    # Reshape output so it's a bunch of rows, one output row for each step for each batch
    seq_output = tf.concat(outputs, axis=1)
    output = tf.reshape(seq_output, [-1, lstm_size])
    
    # Now connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))
    
    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and batch
    logits = tf.matmul(output, softmax_w) + softmax_b
    
    # Use softmax to get the probabilities for predicted characters
    preds = tf.nn.softmax(logits, name='predictions')
    
    # Reshape the targets to match the logits
    y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    cost = tf.reduce_mean(loss)

    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    # Export the nodes
    # NOTE: I'm using a namedtuple here because I think they are cool
    export_nodes = ['inputs', 'targets', 'initial_state', 'final_state',
                    'keep_prob', 'cost', 'preds', 'optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph
    
batch_size = 100
num_steps = 50 
lstm_size = 512
num_layers = 2
learning_rate = 0.001
keep_prob = 0.5

epochs = 20
# Save every N iterations
save_every_n = 1000
train_x, train_y, val_x, val_y = split_data(chars, batch_size, num_steps)

def train():
    model = build_rnn(len(vocab), 
                      batch_size=batch_size,
                      num_steps=num_steps,
                      learning_rate=learning_rate,
                      lstm_size=lstm_size,
                      num_layers=num_layers)
    
    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        # Use the line below to load a checkpoint and resume training
        #saver.restore(sess, 'checkpoints/______.ckpt')
        
        n_batches = int(train_x.shape[1]/num_steps)
        iterations = n_batches * epochs
        for e in range(epochs):
            
            # Train network
            new_state = sess.run(model.initial_state)
            loss = 0
            for b, (x, y) in enumerate(get_batch([train_x, train_y], num_steps), 1):
                iteration = e*n_batches + b
                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([model.cost, model.final_state, model.optimizer], 
                                                     feed_dict=feed)
                loss += batch_loss
                end = time.time()
                print('Epoch {}/{} '.format(e+1, epochs),
                      'Iteration {}/{}'.format(iteration, iterations),
                      'Training loss: {:.4f}'.format(loss/b),
                      '{:.4f} sec/batch'.format((end-start)))
            
                
                if (iteration%save_every_n == 0) or (iteration == iterations):
                    # Check performance, notice dropout has been set to 1
                    val_loss = []
                    new_state = sess.run(model.initial_state)
                    for x, y in get_batch([val_x, val_y], num_steps):
                        feed = {model.inputs: x,
                                model.targets: y,
                                model.keep_prob: 1.,
                                model.initial_state: new_state}
                        batch_loss, new_state = sess.run([model.cost, model.final_state], feed_dict=feed)
                        val_loss.append(batch_loss)
    
                    print('Validation loss:', np.mean(val_loss),
                          'Saving checkpoint!')
                    saver.save(sess, "checkpoints/i{}_l{}_v{:.3f}.ckpt".format(iteration, lstm_size, np.mean(val_loss)))
                    
def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
    
def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    model = build_rnn(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab), 1)
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)

if __name__ == '__main__':   
    if sys.argv[1] == 'train':
        train()
    else:
        checkpoint = "checkpoints/i5000_l512_v1.135.ckpt"
        samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="Man ")
        print(samp)
