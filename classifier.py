import tensorflow as tf
import os
project_dir = os.path.dirname(os.path.realpath(__file__))
data_path = project_dir + "/spiral-train.csv"
tb_summary_path = project_dir + "/tb-summary/"

tf.set_random_seed(0)

def parse_csv(line):
  defaults = [[0.], [0.], [0.]]  # sets field types
  parsed_line = tf.decode_csv(line, defaults)
  # First 3 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(2,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features, label

train_dataset = tf.data.TextLineDataset(data_path).skip(1)
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(100)
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(50)


# create the initialisation operations
#train_init_op = iter.make_initializer(train_dataset)

iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
features, labels = iter.get_next()
train_dataset_init_op = iter.make_initializer(train_dataset)

# Generate tensorflow graph
d = 2
n_hidden = 15

with tf.name_scope("layer-1"):
  W = tf.Variable(tf.random_normal((d, n_hidden)))
  b = tf.Variable(tf.random_normal((n_hidden,)))
  x_1 = tf.nn.relu(tf.matmul(features, W) + b)
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
  entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=labels)
  # Sum all contributions
  l = tf.reduce_sum(entropy)

with tf.name_scope("optim"):
  train_op = tf.train.AdamOptimizer(.001).minimize(l)

with tf.name_scope("summaries"):
  tf.summary.scalar("loss", l)
  merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(tb_summary_path, tf.get_default_graph())

n_steps = 200

with tf.Session() as sess:
   # Train model
   for i in range(n_steps):

      sess.run(train_dataset_init_op)
      sess.run(tf.global_variables_initializer())
      _, summary, loss = sess.run([train_op, merged, l])
      print("step %d, loss: %f" % (i, loss))
      train_writer.add_summary(summary, i)

  # Make Predictions
  #y_pred_np = sess.run(y_pred, feed_dict={x: x_np})

#score = accuracy_score(y_np, y_pred_np)
#print("Classification Accuracy: %f" % score)
