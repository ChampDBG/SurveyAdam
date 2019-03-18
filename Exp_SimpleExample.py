# Linux without GUI
import matplotlib as mpl
mpl.use('Agg')

# packages
import tensorflow as tf
import SelfOptimizer as SO
import matplotlib.pyplot as plt

## Test optimizer
def testing(optimizer_name, iteration = 1000000, learning_rate = 0.001):
    tf.reset_default_graph()

    with tf.Session() as sess:
        r = tf.distributions.Bernoulli(probs = 0.01, dtype = tf.float32).sample()
        x = tf.Variable(0.0)
        loss = (r * 1010 - (1 - r) * 10) * x
        var_list = [x]
        gradient = tf.gradients(loss, var_list)

        if optimizer_name == 'adam':
            optimizer = SO.AdamOptimizer(var_list = var_list, learning_rate = learning_rate)
            print('Using Adam for optimizing.')
        elif optimizer_name == 'amsgrad':
            optimizer = SO.AMSGradOptimizer(var_list = var_list, learning_rate = learning_rate)
            print('Using AMSGrad for optimizing.')
        else:
            raise Exception('Unknown optimizer')

        update_op = optimizer.apply_gradients([(grad, var) for grad, var in zip(gradient, var_list)])
        with tf.control_dependencies([update_op]):
            clip_op = x.assign(tf.clip_by_value(x, -1.0, 1.0))

        sess.run(tf.global_variables_initializer())

        results = []

        for i in range(iteration):
            _, cur_x = sess.run([clip_op, x])
            results.append(cur_x)

            if i % 10000 == 0:
                print('Iteration: %d, current x is %2.4f' % (i, cur_x))
        return results

## main function
def main():
    results_adam = testing('adam', iteration = 5000000, learning_rate = 0.5)
    results_amsg = testing('amsgrad', iteration = 5000000, learning_rate = 0.5)

    plt.plot(results_adam, label = 'adam')
    plt.plot(results_amsg, label = 'amsgrad')
    plt.legend(loc = 'best')
    plt.savefig('./img/Adam_vs_AMSGrad_SimpleExample.png')
    plt.close()

##
if __name__ == '__main__':
    main()