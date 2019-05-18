import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from patsy import bs, dmatrix

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

# def H(np1, np2, df=None, degree=3):
#     D1 = bs(np.linspace(0,1, num=np1), df=df, 
#             degree=degree, include_intercept=True)
#     D2 = bs(np.linspace(0,1, num=np2), df=df, 
#             degree=degree, include_intercept=True)
#     hat = D1 @ np.linalg.inv(D1.T @ D1) @ D2.T
#     return tf.constant(hat, dtype='float32')
# def Dinv(n, df=None, degree=3):
#     D1 = bs(np.linspace(0,1, num=n), df=df, 
#             degree=degree, include_intercept=True)
#     Ds = D1 @ np.linalg.inv(D1.T @ D1) 
#     return tf.constant(Ds, dtype='float32')
# def Basis(num_points, df=None, degree=3):
#     D = bs(np.linspace(0,1, num=num_points), df=df, 
#            degree=degree, include_intercept=True)
#     return tf.constant(D.T, dtype='float32')

# XY = []
# for i in range(28):
#     for j in range(28):
#         XY.append([i/27, j/27])
# XY = tf.constant(XY, dtype='float32')


# def pc_loss(X, points):
#     weight = tf.reshape(X,(-1,784))
#     dist = tf.einsum("ud,ntd->nut", XY**2, tf.ones_like(points)) -\
#            2 * tf.einsum("ud,ntd->nut", XY, points) +\
#            tf.einsum("ud,ntd->nut", tf.ones_like(XY), points**2)
# #     weight_dist = tf.cast(tf.reshape(X,(-1,784,1)) < 0.8, dtype='float32') + dist
#     min_dist = tf.reduce_min(dist, axis = 2)
# #     min_rev = tf.reduce_min(weight_dist, axis = 1)
    
#     losses = tf.reduce_mean(tf.reduce_sum(weight * min_dist, axis=1))
# #              tf.reduce_mean(tf.reduce_sum(min_rev, axis=1))
#     return losses 

# def len_loss(X, points):
#     return tf.reduce_mean(tf.reduce_sum(
#                 tf.sqrt(
#                     tf.clip_by_value(
#                         (points[:,:-1,0] - points[:,1:,0])**2 +\
#                         (points[:,:-1,1] - points[:,1:,1])**2,
#                         1e-7, 1e7)
#                 ),axis=1))

# def distance(points, proj):
#     return tf.einsum("nud,ntd->nut", points**2, tf.ones_like(proj)) -\
#            2 * tf.einsum("nud,ntd->nut", points, proj) +\
#            tf.einsum("nud,ntd->nut", tf.ones_like(points), proj**2)


# # def PointsAutoEncoder(df=5):
# #     inputs = layers.Input(shape=[28,28,1])
# #     x = layers.Conv2D(32, 4, activation='relu', padding='same')(inputs)
# #     x = layers.MaxPooling2D((2, 2), padding='same')(x)
# #     x = layers.Conv2D(32, 4, activation='relu', padding='same')(x)
# #     x = layers.MaxPooling2D((2, 2), padding='same')(x)
# #     x = layers.Conv2D(16, 4, activation='relu', padding='same')(x)
# #     encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# #     encoded = tf.reshape(encoded, (-1,16,16,1))
    
# #     points = layers.Conv2D(2, (1,16), activation='sigmoid')(encoded)
# #     outputs = tf.reshape(points, (-1,16,2))
    
# #     model = keras.Model(inputs=inputs, outputs=outputs)
    
# #     return model
# class Projection(layers.Layer):
#     def __init__(self, unit=128, use_bias=False,
#                  name='projection', **kwargs):
#         super(Projection, self).__init__(name=name, **kwargs)
#         self.Lam = layers.Dense(unit, use_bias=False)
#         self.Gam = layers.Dense(unit, use_bias=use_bias)
#         self.unit = unit
#     def build(self, input_shape):
#         self.spec = layers.InputSpec(shape=input_shape)
#     def get_output_shape_at(self, input_shape):
#         return (input_shape[0], input_shape[1], self.unit)
#     def call(self, x):
#         num = self.spec.shape[1]; dim = self.spec.shape[2]
#         Lx = self.Lam(tf.reshape(x, (-1,dim)))
#         Lx = tf.reshape(Lx,(-1,num,self.unit))
#         sx = tf.reduce_mean(x, axis=1)
#         Gx = tf.reshape(self.Gam(sx),(-1,1,self.unit))
#         x = Lx + tf.tile(Gx, (1,num,1))
#         x = layers.Activation('sigmoid')(x)
#         return x
    
class BS(layers.Layer):
    def __init__(self, df=7, degree=3, 
                 name='Basis', **kwargs):
        super(BS, self).__init__(name=name, **kwargs)
        assert df > degree
        knots1 = np.zeros(degree); knots3 = np.ones(degree)
        knots2 = np.linspace(0,1+1e-7, num=df-degree+1)
        self.knots = np.concatenate([knots1,knots2,knots3])
        self.df = df; self.degree=degree
    def build(self, input_shape):
        self.spec = layers.InputSpec(shape=input_shape)
    def get_output_shape_at(self, input_shape):
        return (input_shape[0], input_shape[1], self.df)
    def call(self, ts):
        df, degree = self.df, self.degree; Bp = []
        knots = self.knots
        for i in range(df+degree):
            if knots[i+1] > knots[i]:
                b0i = (ts >= knots[i]) & (ts < knots[i+1])
                b0i = tf.cast(b0i,dtype='float32')
            else:
                b0i = tf.zeros_like(ts)
            Bp.append(b0i)
        Bp = tf.stack(Bp, axis=2)
        for p in range(1,degree+1):
            new_Bp = []
            for i in range(df+degree-p):
                dn1 = (knots[i+p]-knots[i]) + 1e-15
                dn2 = (knots[i+p+1]-knots[i+1]) + 1e-15
                bpi = (ts - knots[i])/dn1 * Bp[:,:,i] +\
                      (knots[i+p+1]-ts)/dn2 * Bp[:,:,i+1]
                new_Bp.append(bpi)
            Bp = tf.stack(new_Bp, axis=2)
        return Bp

# class Sinkhorn(layers.Layer):
#     def __init__(self, temp=1.0, n_samples=1, 
#                  noise_factor=0.0, n_iters=20, squeeze=True):
#         super(Sinkhorn, self).__init__()
#         self.temp = temp
#         self.n_samples = n_samples
#         self.noise_factor = noise_factor
#         self.n_iters = n_iters
#         self.squeeze = squeeze
#     def call(self, log_alpha):
#         n = tf.shape(log_alpha)[1]
#         log_alpha = tf.reshape(log_alpha, [-1, n, n])
#         batch_size = tf.shape(log_alpha)[0]
#         log_alpha_w_noise = tf.tile(log_alpha, [self.n_samples, 1, 1])
#         if self.noise_factor == 0:
#             noise = 0.0
#         else:
#             uinit = tf.random_uniform_initializer(0,1)
#             u = uinit([self.n_samples*batch_size, n, n], dtype=tf.float32)
#             noise = -tf.math.log(-tf.math.log(u + 1e-20) + 1e-20)*self.noise_factor
#         log_alpha_w_noise += noise
#         log_alpha_w_noise /= self.temp
#         log_alpha_aux = log_alpha_w_noise
#         for _ in range(self.n_iters):
#             log_alpha_aux -= tf.reshape(tf.reduce_logsumexp(log_alpha_aux, axis=2), 
#                                         [-1, n, 1])
#             log_alpha_aux -= tf.reshape(tf.reduce_logsumexp(log_alpha_aux, axis=1), 
#                                         [-1, 1, n])
#         sink = tf.exp(log_alpha_aux)
        
#         if self.n_samples > 1 or self.squeeze is False:
#             sink = tf.reshape(sink, [self.n_samples, batch_size, n, n])
#             sink = tf.transpose(sink, [1, 0, 2, 3])
#         return sink


# XY = []
# for i in range(28):
#     for j in range(28):
#         XY.append([i/27, j/27])
# XY = tf.constant(XY, dtype='float32')

# def pc_loss(X, points):
#     weight = tf.reshape(X,(-1,784))
#     dist = tf.einsum("ud,ntd->nut", XY**2, tf.ones_like(points)) -\
#            2 * tf.einsum("ud,ntd->nut", XY, points) +\
#            tf.einsum("ud,ntd->nut", tf.ones_like(XY), points**2)
#     weight_dist = tf.cast(tf.reshape(X,(-1,784,1)) < 0.8, dtype='float32') + dist
#     min_dist = tf.reduce_min(dist, axis = 2)
#     min_rev = tf.reduce_min(weight_dist, axis = 1)
    
#     losses = tf.reduce_mean(tf.reduce_sum(weight * min_dist, axis=1)) +\
#              tf.reduce_mean(tf.reduce_sum(min_rev, axis=1))
#     return losses 

# def H_loss(X, points):
#     proj = tf.tensordot(points,H(16,16,df=5), axes=(1,0))
#     proj = tf.transpose(proj, (0,2,1))
#     dist = tf.einsum("nud,ntd->nut", points**2, tf.ones_like(proj)) -\
#            2 * tf.einsum("nud,ntd->nut", points, proj) +\
#            tf.einsum("nud,ntd->nut", tf.ones_like(points), proj**2)
#     min_dist = tf.reduce_min(dist, axis = 2)
#     min_rev = tf.reduce_min(dist, axis = 1)
#     return tf.reduce_mean(min_dist+min_rev)

# def len_loss(X, points):
#     return tf.reduce_mean(tf.reduce_sum(
#                 tf.sqrt(
#                     tf.clip_by_value(
#                         (points[:,:-1,0] - points[:,1:,0])**2 +\
#                         (points[:,:-1,1] - points[:,1:,1])**2,
#                         1e-7, 1e7)
#                 ),axis=1))

# def ParaAutoEncoder(nunits=128):
#     inputs = layers.Input(shape=[28,28,1])
#     x = tf.reshape(inputs, (-1, 784))
#     x = layers.Dense(1024, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(1024, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.5)(x)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)
#     x = layers.Dense(32, activation="sigmoid")(x)
#     points = tf.reshape(x, (-1,16,2))
    
#     y = tf.reshape(points, (-1,2))
#     y = layers.Dense(256,activation='sigmoid')(y)
#     y = layers.Dense(16)(y)
#     log_alpha = tf.reshape(y, (-1,16,16))
    
#     perm = Sinkhorn(n_samples=1)(log_alpha)
#     perm = tf.transpose(perm, [0,2,1])
#     perm_points = tf.matmul(perm, points)
    
#     proj = tf.tensordot(perm_points,H(16,128,df=6),axes=(1,0))
#     proj = tf.transpose(proj, [0,2,1])
    
#     model = keras.Model(inputs=inputs, outputs=[points,perm_points,proj,proj])
    
#     return model

# def testAutoEncoder(df=7):
#     inputs = layers.Input(shape=[28,28,1])
#     x = tf.reshape(inputs, (-1, 784))
#     x = layers.Dense(1024, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dense(1024, activation="relu")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.5)(x)
#     x = layers.Dense(512, activation="relu")(x)
#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dense(2*32, activation='sigmoid')(x)
#     para = tf.reshape(x, (-1,32,2))
    
#     points = tf.tensordot(para, H(32,128,df=df), axes=(1,0))
#     points = tf.transpose(points, (0,2,1))
    
#     model = keras.Model(inputs=inputs, outputs=[points,points])
    
#     return model

# def pc_loss_para(df=5):
#     def loss(X, para):
#         points = tf.tensordot(para, H(8,128,df=df), axes=(1,0))
#         points = tf.transpose(points, (0,2,1))
#         return pc_loss(X, points)
#     return loss
# def len_loss_para(df=5):
#     def loss(X, para):
#         points = tf.tensordot(para, H(8,128,df=df), axes=(1,0))
#         points = tf.transpose(points, (0,2,1))
#         return len_loss(X, points)
#     return loss