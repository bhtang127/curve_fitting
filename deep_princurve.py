import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from patsy import bs
import math


class PrincipalCurve(keras.Model):
    def __init__(self,sig2=1e-3,section=3,num_points=30, img_shape=128.0,
                 num_filter=64,reg_lam=1e-3,lambs=[1e-4,1e-3,1e-2],
                 learning_rate=1e-4, drop_rate=0.5,
                 clip_low = 1e-5, clip_high = 10,
                 kernel_type = "gauss"):
        super(curve, self).__init__()
        self.img_shape = 128.0
        self.sig2 = tf.constant(sig2,dtype=tf.float32)
        self.eps = 1e-8
        self.section = section
        self.num_filter = num_filter
        self.num_points = num_points
        self.reg_lam = reg_lam
        self.lambs = lambs
        self.lr = learning_rate
        self.dr = drop_rate
        self.kernel = kernel_type
        self.M = tf.constant([[ 1., 0., -3., 2.],
                              [ 0., 0., 3., -2.],
                              [ 0., 1., -2., 1.],
                              [ 0., 0., -1., 1.]], dtype=tf.float32)
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.reg_loss = 0.0
        self.loss_history = []

        self.Reg = self.l2_reg
        self.Conv1 = layers.Conv3D(filters=self.num_filter,
                                   kernel_size=[5,5,5],
                                   padding="same",
                                   kernel_regularizer=self.Reg,
                                   activation=tf.nn.relu6)
        self.BN1 = layers.BatchNormalization()
        self.DownSampling = layers.AveragePooling3D(pool_size=[2,2,2],strides=[2,2,2])
        self.GlobalPool = layers.GlobalAveragePooling3D()
        # self.MaxPool = layers.MaxPool3D(pool_size=[4,4,4],strides=[4,4,4])
        self.MaxPool2 = layers.MaxPool3D(pool_size=[2,2,2],strides=[2,2,2])
        self.Dropout = layers.Dropout(self.dr)
        self.Conv2 = layers.Conv3D(filters=self.num_filter*2,
                                   kernel_size=[5,5,5],
                                   padding="same",
                                   kernel_regularizer=self.Reg,
                                   activation=tf.nn.relu6)
        self.BN2 = layers.BatchNormalization()
        self.Conv3 = layers.Conv3D(filters=self.num_filter*4,
                                   kernel_size=[5,5,5],
                                   padding="same",
                                   kernel_regularizer=self.Reg,
                                   activation=tf.nn.relu6)
        self.BN3 = layers.BatchNormalization()
        self.Conv4 = layers.Conv3D(filters=self.num_filter*8,
                                   kernel_size=[5,5,5],
                                   padding="same",
                                   kernel_regularizer=self.Reg,
                                   activation=tf.nn.relu6)
        self.BN4 = layers.BatchNormalization()
        self.Dense1 = layers.Dense(1024, activation=tf.nn.relu,
                                   kernel_regularizer=self.Reg)
        self.Dense11 = layers.Dense(1024, activation=tf.nn.relu,
                                   kernel_regularizer=self.Reg)
        self.Dense2 = layers.Dense((self.section+1)*3,
                                    kernel_regularizer=self.Reg)
        self.Dense3 = layers.Dense((self.section+1)*3, 
                                    kernel_regularizer=self.Reg)
        
        self.optimizer = keras.optimizers.Adam(self.lr)
        self.loss = self.total_loss
        super(curve, self).build((None,128,128,128,1))

    def reset_reg(self, lam):
        self.reg_loss = 0
        self.reg_lam = lam
    
    def reset_opt(self, lr):
        self.lr = lr
        self.optimizer = keras.optimizers.Adam(self.lr)

    def l2_reg(self, weight):
        loss = self.reg_lam * tf.reduce_sum(weight**2)
        self.reg_loss += loss
        return loss

    def update_sig2(self,sig2):
        self.sig2 = tf.clip_by_value(sig2,self.clip_low,self.clip_high)

    def reconstruction_xyz(self, loc, moment):
        """Using position and moments to reconstruct points on curve"""
        t_value = np.linspace(0,1, num = self.section+1)
        tpoints = np.linspace(0,1, num=self.num_points)
        basis = np.vstack([tpoints**0, tpoints**1, tpoints**2, tpoints**3])
        basis = tf.constant(basis, dtype=tf.float32)

        i = 0
        x_i = tf.gather(loc, [i,i+1], axis=1)
        m_i = tf.gather(moment, [i,i+1], axis=1)
        xm_i = tf.concat([x_i, m_i], 1)
        coeff = tf.matmul(xm_i, self.M)
        curve = tf.matmul(coeff, basis)
        ### consequent intervals
        for i in range(1, self.section):
            x_i = tf.gather(loc, [i,i+1], axis=1)
            m_i = tf.gather(moment, [i,i+1], axis=1)
            xm_i = tf.concat([x_i, m_i], 1)
            coeff_i = tf.matmul(xm_i, self.M)
            curve_i = tf.matmul(coeff_i, basis)
            coeff = tf.concat([coeff, coeff_i], 1)
            curve = tf.concat([curve, curve_i], 1)
        return curve, coeff

    def reconstruction(self, loc, moment):
        x_points, x_coeff = self.reconstruction_xyz(loc[:,:,0], moment[:,:,0])
        y_points, y_coeff = self.reconstruction_xyz(loc[:,:,1], moment[:,:,1])
        z_points, z_coeff = self.reconstruction_xyz(loc[:,:,2], moment[:,:,2])
        return (x_points,y_points,z_points),(x_coeff,y_coeff,z_coeff)

    def kernel_func(self, dist):
        adjust = tf.einsum("nut,n->nut",dist,self.sig2**(-1))
        if self.kernel == "gauss":
            return tf.exp(-0.5 * adjust)
        if self.kernel == "exponential":
            return tf.exp(-0.5 * tf.math.sqrt(adjust))
        if isinstance(self.kernel, int):
            return (1 + adjust)**(-self.kernel)
    
    def kernel_loss(self, points, true_points, weights):
        turep = tf.cast(true_points, dtype=tf.float32)
        turep /= self.img_shape
        weight = tf.cast(weights, dtype=tf.float32)
        points = tf.stack(points, axis=2)
        dist = tf.einsum("nud,ntd->nut", turep**2, tf.ones_like(points)) -\
            2 * tf.einsum("nud,ntd->nut", turep, points) +\
            tf.einsum("nud,ntd->nut", tf.ones_like(turep), points**2)
        kernel = self.kernel_func(dist)
        sum_kernel = tf.reduce_mean(kernel, axis=2)
        dist_kernel = tf.reduce_mean(dist*kernel,axis=2)
        adjust_sum = tf.clip_by_value(sum_kernel,self.eps,1e7)
        adjust_dist = tf.clip_by_value(dist_kernel,10*self.eps,1e7)
        # use to update global sig2
        sig2_new = tf.reduce_sum(
                        weight * adjust_dist/adjust_sum, axis=1
                        ) /\
                   (3*tf.reduce_sum(weight, axis=1))
        # kernel_distance
        loss = tf.reduce_mean(weight * adjust_dist / adjust_sum)
        return loss, sig2_new

    def decode(self, outputs):
        delta_loc = tf.reshape(outputs[:,:((self.section+1)*3)], [-1, self.section+1, 3])
        delta_moment = tf.reshape(outputs[:,((self.section+1)*3):], [-1, self.section+1, 3])
        loc = tf.cumsum(delta_loc, axis=1) + 0.5
        default_moment = tf.concat([tf.reshape(delta_loc[:,1,:],[-1,1,3]), 
                                   delta_loc[:,1:,:]],axis=1) + \
                         tf.concat([delta_loc[:,1:,:],
                                    tf.reshape(delta_loc[:,-1,:],[-1,1,3])],axis=1)
        default_moment /= 2.0
        moment = (1+delta_moment) * default_moment

        points, coeffs = self.reconstruction(loc,moment)
        return points, coeffs, (loc, moment)

    def total_loss(self,P,W,outputs):
        points, _, _ = self.decode(outputs)
        
        lam_l1 = self.lambs[0]; lam_l2 = self.lambs[1]
        losses, _ = self.kernel_loss(points,P,W)

        losses += lam_l1 * tf.reduce_sum(tf.abs(outputs))
        losses += lam_l2 * tf.reduce_sum(outputs**2)
        
        return losses

    def call(self, inputs):
        inputs = self.DownSampling(inputs)
        #### 64*64*64*1
        conv1 = self.Conv1(inputs); batch1 = self.BN1(conv1)
        pool1 = self.MaxPool2(batch1); dropout1 = self.Dropout(pool1)
        #### 32*32*32*filters
        conv2 = self.Conv2(dropout1); batch2 = self.BN2(conv2)
        pool2 = self.MaxPool2(batch2); dropout2 = self.Dropout(pool2)
        #### 16*16*16*(filters*2)
        conv3 = self.Conv3(dropout2); batch3 = self.BN3(conv3)
        pool3 = self.MaxPool2(batch3); dropout3 = self.Dropout(pool3)
        #### 8*8*8*(filters*4)
        conv4 = self.Conv4(dropout3); batch4 = self.BN4(conv4)
        pool4 = self.MaxPool2(batch4); dropout4 = self.Dropout(pool4)
        #### 4*4*4*(filters*8)
        global_pool = self.GlobalPool(dropout4)
        flat = tf.reshape(global_pool, [-1, self.num_filter*8])

        dense = self.Dense1(flat); dropout5 = self.Dropout(dense)
        dense = self.Dense11(dense); dropout5 = self.Dropout(dense)
        
        loc = self.Dense2(dropout5)
        moment = self.Dense3(dropout5)
        outputs = tf.concat([loc,moment],axis=1)
        return outputs

    def predict(self, inputs):
        outputs = self.call(inputs)
        points,coeffs,para = self.decode(outputs)
        xp, yp, zp = points
        return xp.numpy(), yp.numpy(), zp.numpy()

    def train_on_batch(self, X, P, W, verbose=False):
        if tf.shape(self.sig2).numpy().size == 0:
            self.sig2 = self.sig2 * tf.ones_like(X[:,0,0,0,0])
        self.reg_loss = 0
        with tf.GradientTape() as tape:
            outputs = self.call(X)
            points,_,para = self.decode(outputs)
            _,new_sig2 = self.kernel_loss(points,P,W)
            self.update_sig2(new_sig2)
            losses = self.loss(P, W, outputs)
            losses += self.reg_loss
        
        grads = tape.gradient(losses, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        self.loss_history.append(losses.numpy())
        if verbose:
            print("loss: ",losses.numpy(), "sigma^2:", self.sig2.numpy(), 
                  "suggest sigma^2: ", new_sig2.numpy())      
