import tensorflow as tf

class SinkhornDistance(tf.keras.layers.Layer):
    r"""
    Given two empirical measures with n points each with locations x and y,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
    Shape:
        - Input: :math:`(N, \text{in\_features})`, :math:`(N, \text{in\_features})`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, niter):
        super(SinkhornDistance, self).__init__(trainable=False,name="Sinkhorn")
        self.eps = eps
        self.niter = niter

    def call(self, x, y, weight):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]

        batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        # mu = torch.empty(batch_size, x_points, dtype=torch.float,
        #                  requires_grad=False).fill_(1.0 / x_points).squeeze()
        mu = tf.ones_like(x[:,:,0]) / x_points
        # nu = torch.empty(batch_size, y_points, dtype=torch.float,
        #                  requires_grad=False).fill_(1.0 / y_points).squeeze()
        nu = weight

        U = tf.zeros_like(mu)
        V = tf.zeros_like(nu)

        # Sinkhorn iterations
        for i in range(self.niter):
            U = self.eps * (tf.math.log(mu+1e-8) -\
                            self.lse(self.M(C, U, V))) + U
            V = self.eps * (tf.math.log(nu+1e-8) - \
                            self.lse(tf.transpose(self.M(C, U, V), [0,2,1]))) + V

        # Transport plan pi = diag(a)*K*diag(b)
        pi = tf.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = tf.reduce_sum(pi * C, (-2, -1))

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        # return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        return (-C + tf.expand_dims(u,-1) + tf.expand_dims(v,-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        # x_col = x.unsqueeze(-2)
        # y_lin = y.unsqueeze(-3)
        # C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        # return C
        x_col = tf.expand_dims(x,-2)
        y_lin = tf.expand_dims(y,-3)
        C = tf.reduce_sum(tf.abs(x_col - y_lin)**p, -1)
        return C


    @staticmethod
    def lse(A):
        "log-sum-exp"
        # add 10^-6 to prevent NaN
        # result = torch.log(torch.exp(A).sum(-1) + 1e-6)
        # return result
        res = tf.math.log(tf.reduce_sum(tf.exp(A),-1) + 1e-6)
        return res

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1