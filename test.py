import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_mnist(num_training=55000, num_validation=5000, num_test=10000):

    # Load the raw mnist dataset and use appropriate data types and shapes
    mnist = tf.keras.datasets.mnist.load_data()
    (X_train, y_train), (X_test, y_test) = mnist
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Normalize the data: be in 0-1
    X_train = X_train / 255
    X_val = X_val / 255
    X_test = X_test / 255

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
NHW = (0, 1, 2)
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# plt.figure(1)
# plt.subplot(131)
# plt.imshow(X_train[1,:,:], cmap='gray')
# plt.subplot(132)
# plt.imshow(X_train[5,:,:], cmap='gray')
# plt.subplot(133)
# plt.imshow(X_train[7,:,:], cmap='gray')
# plt.show()

## let's first do num_interior
def coding_net(inputs, num_filter=32, num_interior=3):
    """Coding Net for Image to get interior points and moments"""
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    # Input Layer
    input_layer = tf.reshape(inputs, [-1,28,28,1])
    # First Convolution layer
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=num_filter,
        kernel_size=[5, 5],
        padding="same",
        regularizer=regularizer,
        activation=tf.nn.relu6)
    batch_norm1=tf.layers.batch_normalization(conv1,axis=1)
    pool1 = tf.layers.max_pooling2d(inputs=batch_norm1, pool_size=[2, 2], strides=2)
    dropout1 = tf.layers.dropout(inputs=pool1, rate=0.5)
    
    # Second Convolution layer
    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=num_filter*2,
        kernel_size=[5, 5],
        padding="same",
        regularizer=regularizer,
        activation=tf.nn.relu6)
    batch_norm2=tf.layers.batch_normalization(conv2,axis=1)
    pool2 = tf.layers.max_pooling2d(inputs=batch_norm2, pool_size=[2, 2], strides=2)
    dropout2 = tf.layers.dropout(inputs=pool2, rate=0.5)
    
    # Dense layer
    drop2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=drop2_flat, 
                            units=1024, 
                            regularizer=regularizer,
                            activation=tf.nn.relu6)
    dropout3 = tf.layers.dropout(inputs=dense, rate=0.4)
    
    # Output position and moments
    dense2 = tf.layers.dense(inputs=dropout3, 
                             units= (num_interior+2)*6)
    xy_p = tf.reshape(dense2, [-1, (num_interior+2)*3, 2])
    return xy_p


def reconstruction_x(x_p, num_interior=3, num_points=30):
    """Using position and moments to reconstruct points on curve"""
    M = tf.constant([[1,0,0,0],[1,1,1,1],[0,1,0,0],[0,1,2,3]], 
                    dtype=tf.float64)
    M = tf.transpose(tf.linalg.inv(M))
    t_value = np.linspace(0,1, num = num_interior+2)
    ### first interval
    i = 0
    left_point = t_value[i]; right_point = t_value[i+1]
    tpoints = np.linspace(left_point,right_point, num=num_points)
    basis = np.vstack([tpoints**0, tpoints**1, tpoints**2, tpoints**3])
    basis = tf.constant(basis, dtype=tf.float64)
    x_p = tf.cast(x_p, tf.float64)
    x_p_i = tf.gather(x_p,[3*i, 3*i+2, 3*(i+1), 3*(i+1)+1], axis=1)
    
    coeff = tf.matmul(x_p_i, M)
    curve = tf.matmul(coeff, basis)
    ### consequent intervals
    for i in range(1, num_interior+1):
        left_point = t_value[i]; right_point = t_value[i+1]
        tpoints = np.linspace(left_point,right_point, num=num_points)
        basis = np.vstack([tpoints**0, tpoints**1, tpoints**2, tpoints**3])
        x_p_i = tf.gather(x_p,[3*i, 3*i+2, 3*(i+1), 3*(i+1)+1], axis=1)
        coeff_i = tf.matmul(x_p_i, M)
        curve_i = tf.matmul(coeff_i, basis)
        
        coeff = tf.concat([coeff, coeff_i], 1)
        curve = tf.concat([curve, curve_i], 1)
    
    return curve, coeff

def reconstruction(xy_p, num_interior=3, num_points=30):
    x_points, x_coeff = reconstruction_x(xy_p[:,:,0], num_interior, num_points)
    y_points, y_coeff = reconstruction_x(xy_p[:,:,1], num_interior, num_points)
    
    return x_points, y_points, x_coeff, y_coeff


def loss(x_points, y_points, images, sigma=2, A=5):
    locations = np.array([[i,j] for i in range(28) for j in range(28)]) / 27.0
    locations = tf.constant(locations, dtype=tf.float64)
    pixels = tf.transpose(tf.reshape(images, [-1, 28*28]))
    pixels = tf.cast(pixels, dtype=tf.float64)
    points = tf.stack([x_points, y_points], axis=2)
    
    dist = tf.einsum("ki,nti->knt", locations**2, tf.ones_like(points)) -\
           2*tf.einsum("ki,nti->knt", locations, points) +\
           tf.einsum("ki,nti->knt", tf.ones_like(locations), points**2)
    reduced_dist = tf.reduce_sum(tf.exp(-dist/(2*sigma^2)), axis=2)
    kernal_loss = tf.reduce_mean(reduced_dist * (1-A*pixels))
    
    return kernal_loss

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 28, 28])
Images = tf.placeholder(tf.float32, [None, 28, 28])

coding = coding_net(X)
x_points,y_points,dull,dulll = reconstruction(coding)
losses = loss(x_points,y_points,Images)

reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
losses += reg_term

# def run_model(session, predict, loss_val, Xd, yd,
#               epochs=1, batch_size=64, print_every=100,
#               training=None, plot_losses=False):
#     # have tensorflow compute accuracy
#     correct_prediction = tf.equal(tf.argmax(predict,1), y)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
#     # shuffle indicies
#     train_indicies = np.arange(Xd.shape[0])
#     np.random.shuffle(train_indicies)

#     training_now = training is not None
    
#     # setting up variables we want to compute (and optimizing)
#     # if we have a training function, add that to things we compute
#     variables = [mean_loss,correct_prediction,accuracy]
#     if training_now:
#         variables[-1] = training
    
#     # counter 
#     iter_cnt = 0
#     for e in range(epochs):
#         # keep track of losses and accuracy
#         correct = 0
#         losses = []
#         # make sure we iterate over the dataset once
#         for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
#             # generate indicies for the batch
#             start_idx = (i*batch_size)%Xd.shape[0]
#             idx = train_indicies[start_idx:start_idx+batch_size]
            
#             # create a feed dictionary for this batch
#             feed_dict = {X: Xd[idx,:],
#                          y: yd[idx],
#                          is_training: training_now }
#             # get batch size
#             actual_batch_size = yd[idx].shape[0]
            
#             # have tensorflow compute loss and correct predictions
#             # and (if given) perform a training step
#             loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
#             # aggregate performance stats
#             losses.append(loss*actual_batch_size)
#             correct += np.sum(corr)
            
#             # print every now and then
#             if training_now and (iter_cnt % print_every) == 0:
#                 print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
#                       .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
#             iter_cnt += 1
#         total_correct = correct/Xd.shape[0]
#         total_loss = np.sum(losses)/Xd.shape[0]
#         print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
#               .format(total_loss,total_correct,e+1))
#         if plot_losses:
#             plt.plot(losses)
#             plt.grid(True)
#             plt.title('Epoch {} Loss'.format(e+1))
#             plt.xlabel('minibatch number')
#             plt.ylabel('minibatch loss')
#             plt.show()
#     return total_loss,total_correct

with tf.Session() as sess:
    with tf.device("/gpu:0"): #"/cpu:0" or "/gpu:0" 
        # sess.run(tf.global_variables_initializer())
        # print('Training')
        # run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        # print('Validation')
        # run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
        tf.global_variables_initializer().run()

        losss = sess.run(losses,feed_dict={X:X_train[0:100,:,:],\
                                           Images:X_train[0:100,:,:]})
        print(losss)
