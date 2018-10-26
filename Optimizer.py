import numpy as np
import logging


class ADAGRAD(object):
    def __init__(self, params, lr, lr_word_vector=0.1, epsilon=1e-10):
        logging.info('Optimizer ADAGRAD lr %f' % (lr,))
        self.lr = lr
        self.lr_word_vector = lr_word_vector
        self.epsilon = epsilon
        self.acc_grad = {}
        for param in params:
            self.acc_grad[param] = np.zeros_like(param.get_value())

    def iterate(self, grads):
        lr = self.lr
        epsilon = self.epsilon
        for param, grad in grads.iteritems():
            if param.name == 'Vw':
                param.set_value(param.get_value() - grad.get_value() * self.lr_word_vector)
            else:
                self.acc_grad[param] = self.acc_grad[param] + grad.get_value() ** 2
                param_update = lr * grad.get_value() / (np.sqrt(self.acc_grad[param]) + epsilon)
                param.set_value(param.get_value() - param_update)
            if param.name == 'a_left':
                print("self.a_left ", param.get_value())
            if param.name == 'a_right':
                print("self.a_right ", param.get_value())
            if param.name == 'a':
                print("self.a ", param.get_value())
            if param.name == 'a_in':
                print("self.a_in ", param.get_value())
            if param.name == 'a_out':
                print("self.a_out ", param.get_value())
            if param.name == 'a_aspect_left':
                print("self.a_aspect_left ", param.get_value())
            if param.name == 'a_aspect_right':
                print("self.a_aspect_right ", param.get_value())
            if param.name == 'a_for_left':
                print("self.a_for_left ", param.get_value())
            if param.name == 'a_for_right':
                print("self.a_for_right ", param.get_value())
            if param.name == 'a_back_left':
                print("self.a_back_left ", param.get_value())
            if param.name == 'a_back_right':
                print("self.a_back_right ", param.get_value())
            if param.name == 'b_for_left':
                print("self.b_for_left ", param.get_value())
            if param.name == 'b_for_right':
                print("self.b_for_right ", param.get_value())
            if param.name == 'b_back_left':
                print("self.b_back_left ", param.get_value())
            if param.name == 'b_back_right':
                print("self.b_back_right ", param.get_value())
            if param.name == 'a_for_middle':
                print("self.a_for_middle ", param.get_value())


OptimizerList = {'ADAGRAD': ADAGRAD}
