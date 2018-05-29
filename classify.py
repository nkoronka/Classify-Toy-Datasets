import tensorflow as tf
import os

project_dir = os.path.dirname(os.path.realpath(__file__))
tb_summary_path = project_dir + "/tb-summary/" # command: tensorboard --logdir={path_to_log}

# simple toy dataset from https://cs.joensuu.fi/sipu/datasets/
# format:
#   x,y,class
#   24.75,24.35,2

test_data = project_dir + "/spiral-test.csv"
train_data = project_dir + "/spiral-train.csv"

# Metadata describing the text columns
COLUMNS = ['X', 'Y', 'label']
FIELD_DEFAULTS = [[0.0], [0.0], [1]]
def _parse_line(line): # used for transforming data ahead of training
    # Decode the line into its fields
    fields = tf.decode_csv(line, FIELD_DEFAULTS)
    # Pack the result into a dictionary
    features = dict(zip(COLUMNS,fields))
    # Separate the label from the features
    label = features.pop('label')
    return features, label

train_ds = tf.data.TextLineDataset(train_data).skip(1)
train_ds = train_ds.map(_parse_line)
test_ds = tf.data.TextLineDataset(test_data).skip(1)
test_ds = test_ds.map(_parse_line)
iterator = train_ds.make_initializable_iterator()
features, labels = iterator.get_next(name='iter_next')

# Generate tensorflow graph
dim = 2 # number of input dimensions
n_hidden = 2

with tf.name_scope("placeholders"):
  x = tf.placeholder(tf.float32, (None, dim))
  y = tf.placeholder(tf.float32, (None,))

with tf.name_scope("layer-1"):
  W = tf.Variable(tf.random_normal((dim, n_hidden)))
  b = tf.Variable(tf.random_normal((n_hidden,)))
  x_1 = tf.nn.relu(tf.matmul(x, W) + b)

with tf.name_scope("output"):
  W = tf.Variable(tf.random_normal((n_hidden, 1)))
  b = tf.Variable(tf.random_normal((1,)))
  y_logit = tf.squeeze(tf.matmul(x_1, W) + b)
  # the sigmoid gives the class probability of 1
  y_one_prob = tf.sigmoid(y_logit)
  # Rounding P(y=1) will give the correct prediction.
  y_pred = tf.round(y_one_prob)


with tf.name_scope("loss"):
  # Compute the cross-entropy term for each datapoint
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y)
  # Sum all contributions
  l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
  optimiser = tf.train.AdamOptimizer(.001).minimize(l)

with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  loss_summary = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(tb_summary_path, tf.get_default_graph())
graph = tf.get_default_graph()

n_steps = 200
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(n_steps):

    feed_dict = {x: graph.get_tensor_by_name('iter_next:0')[0], y: graph.get_tensor_by_name('iter_next:0')[1]}
    # feed_dict: A dictionary that maps graph elements to values (described above).

    #_, summary, loss = sess.run([optimiser, loss_summary, l], feed_dict=feed_dict)
    #print("step %d, loss: %f" % (i, loss))
    #train_writer.add_summary(summary, i)

  # Make Predictions
  #y_pred_np = sess.run(y_pred, feed_dict={x: x_np})

#score = accuracy_score(y_np, y_pred_np)
#print("Classification Accuracy: %f" % score)
