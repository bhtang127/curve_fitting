import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from patsy import bs
from Sinkhorn import *

class curve2D(keras.Model):
    def __init__(self, section=3, num_points=30, img_shape=28,
                 num_filter=32,reg_lam=1e-3,lambs=[1e-4,1e-3],
                 learning_rate=1e-4, drop_rate=0.5,
                 clip_low = 1e-5, clip_high = 10,
                 method = "PC", alpha = 0, div = 0.1, niter=50):
        super(curve2D, self).__init__()
        self.img_shape = img_shape
        self.eps = 1e-8
        self.section = section
        self.num_filter = num_filter
        self.num_points = num_points
        self.reg_lam = reg_lam
        self.lambs = lambs
        self.lr = learning_rate
        self.dr = drop_rate
        self.method = method
        self.alpha = alpha
        self.M = tf.constant([[ 1., 0., -3., 2.],
                              [ 0., 0., 3., -2.],
                              [ 0., 1., -2., 1.],
                              [ 0., 0., -1., 1.]], dtype=tf.float32)
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.loss_history = []

        self.Conv1 = layers.Conv2D(filters=self.num_filter,
                                   kernel_size=[4,4],
                                   padding="same",
                                   activation=tf.nn.relu6)
        self.BN1 = layers.BatchNormalization()
        self.MaxPool = layers.MaxPool2D(pool_size=[2,2],strides=[2,2])
        self.Dropout = layers.Dropout(self.dr)
        self.Conv2 = layers.Conv2D(filters=self.num_filter*2,
                                   kernel_size=[4,4],
                                   padding="same",
                                   activation=tf.nn.relu6)
        self.BN2 = layers.BatchNormalization()
        
        self.Dense1 = layers.Dense(1024, activation=tf.nn.relu)
        self.Dense2 = layers.Dense((self.section+1)*2, activation=tf.nn.sigmoid)
        self.Dense3 = layers.Dense((self.section+1)*2, use_bias=False)
        
        self.optimizer = keras.optimizers.Adam(self.lr)
        self.loss = self.total_loss
        self.div = div
        self.niter = niter
        self.sinkhorn = SinkhornDistance(self.div, self.niter)
        self.sinkhorn.build([1,28,28,1])
        super(curve2D, self).build((None,28,28,1))
    
    def reset_sinkhorn(self, div, niter=50):
        self.div = div
        self.niter = niter
        self.sinkhorn = SinkhornDistance(self.div, self.niter)
        self.sinkhorn.build([1,28,28,1])

    def reset_opt(self, lr):
        self.lr = lr
        self.optimizer = keras.optimizers.Adam(self.lr)

    def decrease_learning_rate(self, factor):
        self.lr /= factor
        self.optimizer = keras.optimizers.Adam(self.lr)
    
    def decrease_len_penalty(self, factor):
        self.lambs[0] /= factor
    
    def decrease_smooth_penalty(self, factor):
        self.lambs[1] /= factor

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
        return (x_points,y_points),(x_coeff,y_coeff)
    
    def kernel_loss(self, points, true_points, weights):
        turep = tf.cast(true_points, dtype=tf.float32)
        turep /= self.img_shape
        weight = tf.cast(weights, dtype=tf.float32)
        weight /= tf.reduce_sum(weight)
        points = tf.stack(points, axis=2)

        loss = 0

        if self.method == "PC":
            dist = tf.einsum("nud,ntd->nut", turep**2, tf.ones_like(points)) -\
                2 * tf.einsum("nud,ntd->nut", turep, points) +\
                tf.einsum("nud,ntd->nut", tf.ones_like(turep), points**2)
            
            min_dist = tf.reduce_min(dist, axis = 2)
            min_rev = tf.reduce_min(dist, axis = 1)

            loss += tf.reduce_mean(tf.reduce_sum(weight * min_dist, axis=1)) + \
                self.alpha * tf.reduce_mean(tf.reduce_sum(min_rev, axis=1))

        if self.method == "Wass":
            losses,_,_ = self.sinkhorn(points, turep, weight)
            loss += tf.reduce_sum(losses)
    
        return loss


    def decode(self, outputs):
        loc = tf.reshape(outputs[:,:((self.section+1)*2)], [-1, self.section+1, 2])
        delta_loc = loc[:,1:,:] - loc[:,:-1,:]
        delta_moment = tf.reshape(outputs[:,((self.section+1)*2):], [-1, self.section+1, 2])
        default_moment = tf.concat([tf.reshape(delta_loc[:,0,:],[-1,1,2]), 
                                   delta_loc],axis=1) + \
                         tf.concat([delta_loc,
                                    tf.reshape(delta_loc[:,-1,:],[-1,1,2])],axis=1)
        default_moment /= 2.0
        moment = delta_moment + default_moment

        points, coeffs = self.reconstruction(loc,moment)
        return points, coeffs, (loc, moment)

    def total_loss(self,P,W,outputs):
        points, _, _ = self.decode(outputs)
        xpo, ypo = points
        
        lam_len = self.lambs[0]; lam_moment = self.lambs[1]
        losses = self.kernel_loss(points,P,W)
        
        delta_moment = tf.reshape(outputs[:,((self.section+1)*2):], [-1, self.section+1, 2])
        
        losses += lam_len * tf.reduce_mean(
                                tf.reduce_sum(
                                    tf.sqrt(
                                        tf.clip_by_value(
                                            (xpo[:,:-1] - xpo[:,1:])**2 +\
                                            (ypo[:,:-1] - ypo[:,1:])**2,
                                            self.eps, 1/self.eps)
                                        ),
                                    axis=1
                                )
                            )
        losses += lam_moment * tf.reduce_mean(
                                    tf.reduce_sum(delta_moment**2,axis=[1,2])
                               )
        
        return losses

    def call(self, inputs):
        #### 28*28*1
        conv1 = self.Conv1(inputs); batch1 = self.BN1(conv1)
        pool1 = self.MaxPool(batch1); dropout = self.Dropout(pool1)
        #### 14*14*filters
        conv2 = self.Conv2(dropout); batch2 = self.BN2(conv2)
        pool2 = self.MaxPool(batch2); dropout = self.Dropout(pool2)
        #### 7*7*(filters*2)
        flat = tf.reshape(dropout, [-1, self.num_filter*2*7*7])

        dense = self.Dense1(flat)
        dropout2 = self.Dropout(dense)
        
        loc = self.Dense2(dropout2)
        moment = self.Dense3(dropout2)
        outputs = tf.concat([loc,moment],axis=1)
        return outputs

    def predict(self, inputs):
        outputs = self.call(inputs)
        points,coeffs,para = self.decode(outputs)
        xp, yp = points
        return xp.numpy(), yp.numpy()

    def train_on_batch(self, X, P, W, verbose=False):
        with tf.GradientTape() as tape:
            outputs = self.call(X)
            points,_,para = self.decode(outputs)
            losses = self.loss(P, W, outputs)
        
        grads = tape.gradient(losses, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        self.loss_history.append(losses.numpy())



# class curve2D(keras.Model):
#     def __init__(self,sig2=1e-3,section=3,num_points=30, img_shape=28,
#                  num_filter=64,reg_lam=1e-3,lambs=[1e-4,1e-3],
#                  learning_rate=1e-4, drop_rate=0.5,
#                  clip_low = 1e-5, clip_high = 10,
#                  kernel_type = "gauss"):
#         super(curve2D, self).__init__()
#         self.img_shape = img_shape
#         self.sig2 = tf.constant(sig2,dtype=tf.float32)
#         self.eps = 1e-8
#         self.section = section
#         self.num_filter = num_filter
#         self.num_points = num_points
#         self.reg_lam = reg_lam
#         self.lambs = lambs
#         self.lr = learning_rate
#         self.dr = drop_rate
#         self.kernel = kernel_type
#         self.M = tf.constant([[ 1., 0., -3., 2.],
#                               [ 0., 0., 3., -2.],
#                               [ 0., 1., -2., 1.],
#                               [ 0., 0., -1., 1.]], dtype=tf.float32)
#         self.clip_low = clip_low
#         self.clip_high = clip_high
#         self.loss_history = []

#         self.Conv1 = layers.Conv2D(filters=self.num_filter,
#                                    kernel_size=[4,4],
#                                    padding="same",
#                                    activation=tf.nn.relu6)
#         self.BN1 = layers.BatchNormalization()
#         self.MaxPool = layers.MaxPool2D(pool_size=[2,2],strides=[2,2])
#         self.Dropout = layers.Dropout(self.dr)
#         self.Conv2 = layers.Conv2D(filters=self.num_filter*2,
#                                    kernel_size=[4,4],
#                                    padding="same",
#                                    activation=tf.nn.relu6)
#         self.BN2 = layers.BatchNormalization()
        
#         self.Dense1 = layers.Dense(1024, activation=tf.nn.relu)
#         self.Dense2 = layers.Dense((self.section+1)*2, activation=tf.nn.sigmoid)
#         self.Dense3 = layers.Dense((self.section+1)*2, use_bias=False)
        
#         self.optimizer = keras.optimizers.Adam(self.lr)
#         self.loss = self.total_loss

#         self.index_tensor = np.zeros([28,28,2])
#         for i in range(28):
#             for j in range(28):
#                 self.index_tensor[i,j,:] = [i/27.0, j/27.0]
#         self.index_tensor = tf.constant(self.index_tensor, dtype=tf.float32)
#         super(curve2D, self).build((None,28,28,1))
    
#     def reset_opt(self, lr):
#         self.lr = lr
#         self.optimizer = keras.optimizers.Adam(self.lr)

#     def reconstruction_xyz(self, loc, moment):
#         """Using position and moments to reconstruct points on curve"""
#         t_value = np.linspace(0,1, num = self.section+1)
#         tpoints = np.linspace(0,1, num=self.num_points)
#         basis = np.vstack([tpoints**0, tpoints**1, tpoints**2, tpoints**3])
#         basis = tf.constant(basis, dtype=tf.float32)

#         i = 0
#         x_i = tf.gather(loc, [i,i+1], axis=1)
#         m_i = tf.gather(moment, [i,i+1], axis=1)
#         xm_i = tf.concat([x_i, m_i], 1)
#         coeff = tf.matmul(xm_i, self.M)
#         curve = tf.matmul(coeff, basis)
#         ### consequent intervals
#         for i in range(1, self.section):
#             x_i = tf.gather(loc, [i,i+1], axis=1)
#             m_i = tf.gather(moment, [i,i+1], axis=1)
#             xm_i = tf.concat([x_i, m_i], 1)
#             coeff_i = tf.matmul(xm_i, self.M)
#             curve_i = tf.matmul(coeff_i, basis)
#             coeff = tf.concat([coeff, coeff_i], 1)
#             curve = tf.concat([curve, curve_i], 1)
#         return curve, coeff

#     def reconstruction(self, loc, moment):
#         x_points, x_coeff = self.reconstruction_xyz(loc[:,:,0], moment[:,:,0])
#         y_points, y_coeff = self.reconstruction_xyz(loc[:,:,1], moment[:,:,1])
#         return (x_points,y_points),(x_coeff,y_coeff)

#     def kernel_func(self, adjust):
#         if self.kernel == "gauss" or self.kernel == "min":
#             return tf.exp(-0.5 * adjust)
#         if self.kernel == "exponential":
#             return tf.exp(- tf.math.sqrt(adjust))
#         if isinstance(self.kernel, int):
#             return (1 + adjust)**(-self.kernel)
    
#     def kernel_sigma(self, points, true_points, weights):
#         turep = tf.cast(true_points, dtype=tf.float32)
#         turep /= self.img_shape
#         weight = tf.cast(weights, dtype=tf.float32)
#         points = tf.stack(points, axis=2)
        
#         dist = tf.einsum("nud,ntd->nut", turep**2, tf.ones_like(points)) -\
#             2 * tf.einsum("nud,ntd->nut", turep, points) +\
#             tf.einsum("nud,ntd->nut", tf.ones_like(turep), points**2)
        
#         ## iterate for sig2
#         adjust = dist / self.sig2
#         kernel = self.kernel_func(adjust)
#         adjust_kernel = tf.clip_by_value(kernel,self.eps,1e7)
#         kernel_dist = tf.reduce_sum(dist*adjust_kernel, axis=1)
#         tw = 2*tf.reduce_sum(adjust_kernel, axis=1)
#         sig2 = kernel_dist / tw
#         sig2 = tf.clip_by_value(sig2,self.eps,1e7)
        
#         adjust = tf.einsum("nut,nt->nut", dist, sig2**(-1))
#         kernel = self.kernel_func(adjust)
#         adjust_kernel = tf.clip_by_value(kernel,self.eps,1e7)
#         kernel_dist = tf.reduce_sum(dist*adjust_kernel, axis=1)
#         tw = 2*tf.reduce_sum(adjust_kernel, axis=1)
#         sig2 = kernel_dist / tw
#         sig2 = tf.clip_by_value(sig2,self.eps,1e7)
        
#         adjust = tf.einsum("nut,nt->nut", dist, sig2**(-1))
#         kernel = self.kernel_func(adjust)
#         adjust_kernel = tf.clip_by_value(kernel,self.eps,1e7)
#         kernel_dist = tf.reduce_sum(dist*adjust_kernel, axis=1)
#         tw = 2*tf.reduce_sum(adjust_kernel, axis=1)
#         sig2 = kernel_dist / tw
#         sig2 = tf.clip_by_value(sig2,self.eps,1e7)

#         return sig2

#     def kernel_image(self, points, sig2):
#         points = tf.stack(points, axis=2)
#         dist = tf.einsum("iju,ntu->nijt", self.index_tensor**2, tf.ones_like(points)) -\
#             2 * tf.einsum("iju,ntu->nijt", self.index_tensor, points) +\
#             tf.einsum("iju,ntu->nijt", tf.ones_like(self.index_tensor), points**2)
#         adjust = tf.einsum("nijt,nt->nijt", dist, sig2**(-1))
#         kernel = self.kernel_func(adjust)
#         images = tf.einsum("nijt,nt->nij", kernel, sig2**(-1))
#         W = tf.reduce_sum(images, axis=[1,2])
#         images = tf.einsum("nij,n->nij", images, W**(-1))
        
#         return images

#     def kernel_loss(self, kernel_img, images):
#         images = tf.constant(images[:,:,:,0], dtype=tf.float32)
#         W = tf.reduce_sum(images, axis=[1,2])
#         images = tf.einsum("nij,n->nij", images, W**(-1))
#         return tf.reduce_sum( (kernel_img - images)**2 )


#     def decode(self, outputs):
#         loc = tf.reshape(outputs[:,:((self.section+1)*2)], [-1, self.section+1, 2])
#         delta_loc = loc[:,1:,:] - loc[:,:-1,:]
#         delta_moment = tf.reshape(outputs[:,((self.section+1)*2):], [-1, self.section+1, 2])
#         default_moment = tf.concat([tf.reshape(delta_loc[:,0,:],[-1,1,2]), 
#                                    delta_loc],axis=1) + \
#                          tf.concat([delta_loc,
#                                     tf.reshape(delta_loc[:,-1,:],[-1,1,2])],axis=1)
#         default_moment /= 2.0
#         moment = delta_moment + default_moment

#         points, coeffs = self.reconstruction(loc,moment)
#         return points, coeffs, (loc, moment)

#     def total_loss(self,P,W,X,outputs):
#         points, _, _ = self.decode(outputs)
#         xpo, ypo = points
        
#         lam_len = self.lambs[0]; lam_moment = self.lambs[1]
#         sig2 = self.kernel_sigma(points,P,W)
#         images = self.kernel_image(points,sig2)
#         losses = self.kernel_loss(images, X)
        
#         delta_moment = tf.reshape(outputs[:,((self.section+1)*2):], [-1, self.section+1, 2])
        
#         losses += lam_len * tf.reduce_mean(
#                                 tf.reduce_sum(
#                                     tf.sqrt(
#                                         tf.clip_by_value(
#                                             (xpo[:,:-1] - xpo[:,1:])**2 +\
#                                             (ypo[:,:-1] - ypo[:,1:])**2,
#                                             self.eps, 1/self.eps)
#                                         ),
#                                     axis=1
#                                 )
#                             )
#         losses += lam_moment * tf.reduce_mean(
#                                     tf.reduce_sum(delta_moment**2,axis=[1,2])
#                                )
        
#         return losses

#     def call(self, inputs):
#         #### 28*28*1
#         conv1 = self.Conv1(inputs); batch1 = self.BN1(conv1)
#         pool1 = self.MaxPool(batch1); dropout = self.Dropout(pool1)
#         #### 14*14*filters
#         conv2 = self.Conv2(dropout); batch2 = self.BN2(conv2)
#         pool2 = self.MaxPool(batch2); dropout = self.Dropout(pool2)
#         #### 7*7*(filters*2)
#         flat = tf.reshape(dropout, [-1, self.num_filter*2*7*7])

#         dense = self.Dense1(flat)
#         dropout2 = self.Dropout(dense)
        
#         loc = self.Dense2(dropout2)
#         moment = self.Dense3(dropout2)
#         outputs = tf.concat([loc,moment],axis=1)
#         return outputs

#     def predict(self, inputs):
#         outputs = self.call(inputs)
#         points,coeffs,para = self.decode(outputs)
#         xp, yp = points
#         return xp.numpy(), yp.numpy()

#     def train_on_batch(self, X, P, W, verbose=False):
#         with tf.GradientTape() as tape:
#             outputs = self.call(X)
#             points,_,para = self.decode(outputs)
#             losses = self.loss(P, W, X, outputs)
        
#         grads = tape.gradient(losses, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
#         self.loss_history.append(losses.numpy())