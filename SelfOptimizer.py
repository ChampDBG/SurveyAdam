# packages
import tensorflow as tf

class AdamOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8, var_list = []):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.var_list = var_list
        self.m = {}
        self.v = {}
        self.t = tf.Variable(0.0, trainable = False)
        # for bias correction
        self.hat_beta1 = tf.Variable(1.0, trainable = False)
        self.hat_beta2 = tf.Variable(1.0, trainable = False)

        for var in self.var_list:
            var_shape = tf.shape(var.initial_value)
            self.m[var] = tf.Variable(tf.zeros(var_shape), trainable = False)
            self.v[var] = tf.Variable(tf.zeros(var_shape), trainable = False)

    def apply_gradients(self, gradient_variables):
        ## bias correct
        # beta1 and beta2 for bias correction in this time
        beta1_t = self.hat_beta1.assign(self.hat_beta1 * self.beta1)
        beta2_t = self.hat_beta2.assign(self.hat_beta2 * self.beta2)

        # calculate
        with tf.control_dependencies([self.t.assign_add(1.0), beta1_t, beta2_t]):
            learning_rate = self.learning_rate / tf.sqrt(self.t)
            update_ops = []

            for (g, var) in gradient_variables:
                m = self.m[var].assign(self.beta1 * self.m[var] + (1-self.beta1) * g)
                v = self.v[var].assign(self.beta2 * self.v[var] + (1-self.beta2) * g * g)
                m_hat = m / (1 - beta1_t)
                v_hat = m / (1 - beta2_t)

                update = -learning_rate * m / (self.epsilon + tf.sqrt(v_hat))
                update_ops.append(var.assign_add(update))
            return tf.group(*update_ops)


class AMSGradOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8, var_list = []):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.var_list = var_list
        self.m = {}
        self.v = {}
        self.v_hat = {}
        self.t = tf.Variable(0.0, trainable = False)

        for var in self.var_list:
            var_shape = tf.shape(var.initial_value)
            self.m[var] = tf.Variable(tf.zeros(var_shape), trainable = False)
            self.v[var] = tf.Variable(tf.zeros(var_shape), trainable = False)
            self.v_hat[var] = tf.Variable(tf.zeros(var_shape), trainable = False)

    def apply_gradients(self, gradient_variables):
        with tf.control_dependencies([self.t.assign_add(1.0)]):
            learning_rate = self.learning_rate / tf.sqrt(self.t)
        update_ops = []

        for (g, var) in gradient_variables:
            m = self.m[var].assign(self.beta1 * self.m[var] + (1 - self.beta1) * g)
            v = self.v[var].assign(self.beta2 * self.v[var] + (1 - self.beta2) * g * g)
            selector = tf.maximum(self.v_hat[var], v)
            v_hat = self.v_hat[var].assign(selector)

            update = -learning_rate * m / (self.epsilon + tf.sqrt(v_hat))
            update_ops.append(var.assign_add(update))

        return tf.group(*update_ops)