import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def autoencoder(inputs,hidden_units):
    hidden=tf.contrib.layers.fully_connected(inputs,hidden_units,activation_fn=tf.nn.sigmoid)
    decoder=tf.contrib.layers.fully_connected(hidden,inputs.shape[1].value,activation_fn=tf.nn.sigmoid)
    return decoder

x=tf.placeholder(tf.float32,[None,784])

x_decoder=autoencoder(x,50)
loss=tf.reduce_mean(tf.squared_difference(x,x_decoder))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        x_batch,_=mnist.train.next_batch(128)
        loss_step,_=sess.run([loss,optimizer],feed_dict={x:x_batch})
        if i%10==0:
            print loss_step

    x_batch_test,_=mnist.train.next_batch(10)
    x_decoder_test=sess.run(x_decoder,feed_dict={x:x_batch_test})
    fig,ax=plt.subplots(2,10)
    for j in range(10):
        ax[0][j].imshow(x_batch_test[j].reshape([28,28]))
        ax[1][j].imshow(x_decoder_test[j].reshape([28,28]))
    fig.show()
    plt.draw()
    plt.savefig('autoencoder.png')
    plt.pause(10)
