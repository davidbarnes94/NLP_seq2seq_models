import tensorflow as tf
import numpy as np

#100 images?

b = tf.Variable(tf.zeros((100, )))

W = tf.Variable(tf.random_uniform((784, 100),-1, 1))

x = tf.placeholder(tf.float32, (100, 784)) #why that shape?

h = tf.nn.relu(tf.matmul(x, W) + b)

sess = tf.Session()
sess.run(tf.initialize_all_variables)
sess.run(h, {x: np.random.random(100, 784)})

prediction = tf.nn.softmax()
label = tf.placeholder(tf.float32, [100, 10])
cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)

##Gradient Calculation
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#minimize function: it complutes the gradient of the argument w.r.t. variables
# ...apply gradient updates to the variables

#expressing the model as a computational graph makes it very advantageous for computing gradients
#parameter for gradient descent optimizer is the learning rate

for i in range(1000):
    batch_x, batch_label = data.next_batch()
    sess.run(train_step, feed_dict={x: batch_x, label: batch_label})
    #data is an arbitrary dataset


