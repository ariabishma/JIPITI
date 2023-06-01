import numpy as np 
import tensorflow as tf

context_size = 8
n_embd = 64
n_head = 4
head_size = 64
n_dropout = .2
minibatch_size = 32

datasets = tf.data.TextLineDataset(['./datasets/puisi_mini.txt']) # Training Dataset
validation = tf.data.TextLineDataset(['./datasets/puisi_mini_validation.txt']) # Training Dataset

vectorizer = tf.keras.layers.TextVectorization(ragged=True)
vectorizer.adapt(datasets.batch(64))

vocab_size = vectorizer.vocabulary_size()
print(vocab_size)

vocabs = vectorizer.get_vocabulary()

decode = lambda encoded : ' '.join([vocabs[i] for i in encoded])

batches = datasets.batch(16).repeat().shuffle(1000, reshuffle_each_iteration=True)

val_batches = validation.batch(16).shuffle(500, reshuffle_each_iteration=True)


iterator = iter(batches)

def get_batch():
    data = vectorizer(iterator.get_next()).flat_values
    ix   = np.random.randint(len(data) - context_size, size = (minibatch_size,)) # generate random text position and random batch sizes
    x    = np.stack([data[i:i+context_size] for i in ix])
    y    = np.stack([data[i+1:i+context_size+1] for i in ix])
    return x, y

def validate(model, loss_fn):
    losses = []
    for v in val_batches:
        data   = vectorizer(v).flat_values
        ix     = np.random.randint(len(data) - context_size, size = (minibatch_size,)) # generate random text position and random batch sizes
        x_val  = np.stack([data[i:i+context_size] for i in ix])
        y_val  = np.stack([data[i+1:i+context_size+1] for i in ix])

        logits = model(x_val)
        loss = loss_fn(y_val, logits)
        losses.append(loss.numpy())

    losses = np.array(losses).mean()
    
    return losses



class SelfAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.query = tf.keras.layers.Dense(head_size, use_bias=False)
        self.key = tf.keras.layers.Dense(head_size, use_bias=False)
        self.value = tf.keras.layers.Dense(head_size, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(n_dropout)

    def call(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        x = query @ tf.transpose(key, perm=[0,2,1])

        mask = tf.linalg.band_part(tf.ones_like(x ,dtype='bool'), -1, 0)
        x = tf.where(mask, x, float('-inf')) # masked
        x = tf.nn.softmax(x / np.sqrt(head_size))

        x = self.dropout(x @ value)

        return x 
    

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = [SelfAttention() for _ in range(n_head)]
        self.projection = tf.keras.layers.Dense(n_embd, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(n_dropout)
        self.layernorm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        heads = []
        
        for h in self.heads:
            heads.append(h(x))

        heads = tf.concat(heads,-1)
        x = x + self.projection(heads)
        x = self.layernorm(x)
        x = self.dropout(x)

        return x
    

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.mid = tf.keras.layers.Dense(n_embd * 4, activation='relu')
        self.head = tf.keras.layers.Dense(n_embd)
        self.layernorm = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        x = self.mid(x)
        x = self.layernorm(self.head(x))
        return x
        

class Block(tf.keras.layers.Layer):
    def __init__(self):
        super(Block, self).__init__()
        self.multiHeadSelfAttention = MultiHeadSelfAttention()
        self.ffn = FullyConnected()
        
    def call(self, x):
        logits = self.multiHeadSelfAttention(logits)
        logits = self.ffn(logits)

        return logits

class LanguageModel(tf.keras.Model):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.pos_embd= tf.keras.layers.Embedding(context_size, n_embd)

        self.multiHeadSelfAttention = MultiHeadSelfAttention()
        self.ffn = FullyConnected()

        self.head  = tf.keras.layers.Dense(vocab_size)

    def call(self, input):
        token_embd = self.embed(input)
        pos_embd = self.pos_embd(tf.range(context_size))
        logits = token_embd + pos_embd

        logits = self.multiHeadSelfAttention(logits)
        logits = self.ffn(logits)
        logits = self.head(logits)

        return logits 
    

model = LanguageModel()

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

for i in range(3000):
    with tf.GradientTape() as tape:
        x, y = get_batch()
        logits = model(x)
        loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    if i % 10 == 0:
        val_loss = validate(model,loss_fn)
        print('Iteration : ',i,' train_loss : ',loss.numpy(),' val_loss : ',val_loss)

print('-----------------Context----------------')

context = 'Ceritakanlah hari yang telah kau lalui Matahari terlihat'
print(context)
context = vectorizer(context)
context = list(context.numpy())

for i in range(100):
    logits = model(np.array([context[-context_size:]]))
    prediction = logits[-1:,-1] # (1, vocab_size)
    prediction = tf.random.categorical(prediction, num_samples=1)
    context.append(prediction[0][0].numpy())

print('-----------------Result----------------')
print(decode(context))
print('---------------------------------------')


model.summary()
# model.save('./jipiti_1000_puisi')
