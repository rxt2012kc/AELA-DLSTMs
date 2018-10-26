# -*- coding: utf8 -*-
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import argparse
import time
import collections
from WordLoader import WordLoader


class AELADLSTMS(object):
    def __init__(self, wordlist, argv, aspect_num=0):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='lstm')
        parser.add_argument('--rseed', type=int, default=int(1000 * time.time()) % 19491001)
        parser.add_argument('--dim_word', type=int, default=300)
        parser.add_argument('--dim_hidden', type=int, default=300)
        parser.add_argument('--dim_aspect', type=int, default=300)
        parser.add_argument('--grained', type=int, default=3, choices=[3])
        parser.add_argument('--regular', type=float, default=0.001)
        parser.add_argument('--word_vector', type=str, default='data/glove.840B.300d.txt')
        args, _ = parser.parse_known_args(argv)

        self.name = args.name
        self.srng = RandomStreams(seed=args.rseed)
        self.dim_word, self.dim_hidden = args.dim_word, args.dim_hidden
        self.dim_aspect = args.dim_aspect
        self.grained = args.grained
        self.regular = args.regular
        self.num = len(wordlist) + 1
        self.init_param()
        self.load_word_vector(args.word_vector, wordlist)
        self.init_function()

    def init_param(self):
        def shared_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX) + b
            f = theano.function([], matrix)
            return theano.shared(f(), name=name)

        u = lambda x: 1 / np.sqrt(x)
        dimc, dimh, dima = self.dim_word, self.dim_hidden, self.dim_aspect
        dim_lstm_para = dimh + dimc
        self.Vw = shared_matrix((self.num, dimc), 'Vw', 0.01)
        self.Wi = shared_matrix((dimh, dim_lstm_para), 'Wi', u(dimh))
        self.Wo = shared_matrix((dimh, dim_lstm_para), 'Wo', u(dimh))
        self.Wf = shared_matrix((dimh, dim_lstm_para), 'Wf', u(dimh))

        self.bi = shared_matrix((dimh,), 'bi', 0.)
        self.bo = shared_matrix((dimh,), 'bo', 0.)
        self.bf = shared_matrix((dimh,), 'bf', 0.)

        self.Wc = shared_matrix((dimh, dim_lstm_para), 'Wc', u(dimh))
        self.bc = shared_matrix((dimh,), 'bc', 0.)

        self.Ws = shared_matrix((dimh + dimh, self.grained), 'Ws', u(dimh))
        self.bs = shared_matrix((self.grained,), 'bs', 0.)

        self.h0, self.c0 = np.zeros(dimh, dtype=theano.config.floatX), np.zeros(dimc,
                                                                                dtype=theano.config.floatX)
        self.params = [self.Wi, self.Wo, self.Wf, self.Wc, self.bi, self.bo, self.bf, self.bc, self.Ws,
                       self.bs]
        self.Wp_L = shared_matrix((dimh, dimh), 'Wp', u(dimh))
        self.Wx_L = shared_matrix((dimh, dimh), 'Wx', u(dimh))
        self.Wp_R = shared_matrix((dimh, dimh), 'Wp', u(dimh))
        self.Wx_R = shared_matrix((dimh, dimh), 'Wx', u(dimh))
        self.params.extend([self.Wp_L, self.Wx_L, self.Wp_R, self.Wx_R])

        self.alpha_h_W_L = shared_matrix((dimh, dimh + dimh), 'alpha_h_W_L', u(dimh * 2))
        self.alpha_h_W_R = shared_matrix((dimh, dimh + dimh), 'alpha_h_W_R', u(dimh * 2))
        self.params.extend([self.alpha_h_W_L, self.alpha_h_W_R])

        self.a_for_left = theano.shared(1.0, name='a_for_left')
        self.a_for_middle = theano.shared(1.0, name='a_for_middle')
        self.b_for_left = theano.shared(0.0, name='b_for_left')

        self.a_back_right = theano.shared(1.0, name='a_back_right')
        self.b_back_right = theano.shared(0.0, name='b_back_right')

        self.params.extend([self.a_for_left, self.a_for_middle, self.b_for_left])
        self.params.extend([self.a_back_right, self.b_back_right])

    def init_function(self):

        self.seq_loc = T.lvector()
        self.seq_idx = T.lvector()
        self.target = T.lvector()
        self.target_content_index = T.lscalar()
        self.seq_len = T.lscalar()
        self.solution = T.matrix()
        self.seq_matrix = T.take(self.Vw, self.seq_idx, axis=0)

        self.all_tar_vector = T.take(self.Vw, self.target, axis=0)
        self.tar_vector = T.mean(self.all_tar_vector, axis=0)
        self.target_vector_dim = self.tar_vector.dimshuffle('x', 0)
        self.seq_matrix = T.concatenate([self.seq_matrix[0:self.target_content_index], self.target_vector_dim,
                                         self.seq_matrix[self.target_content_index + 1:]], axis=0)
        h, c = T.zeros_like(self.bf, dtype=theano.config.floatX), T.zeros_like(self.bc,
                                                                               dtype=theano.config.floatX)

        def rnn(X, aspect):
            def encode_forward(x_t, h_fore, c_fore):
                v = T.concatenate([h_fore, x_t])
                f_t = T.nnet.sigmoid(T.dot(self.Wf, v) + self.bf)
                i_t = T.nnet.sigmoid(T.dot(self.Wi, v) + self.bi)
                o_t = T.nnet.sigmoid(T.dot(self.Wo, v) + self.bo)
                c_next = f_t * c_fore + i_t * T.tanh(T.dot(self.Wc, v) + self.bc)
                h_next = o_t * T.tanh(c_next)
                return h_next, c_next

            def encode_backward(x_t, h_fore, c_fore):
                v = T.concatenate([h_fore, x_t])
                f_t = T.nnet.sigmoid(T.dot(self.Wf, v) + self.bf)
                i_t = T.nnet.sigmoid(T.dot(self.Wi, v) + self.bi)
                o_t = T.nnet.sigmoid(T.dot(self.Wo, v) + self.bo)
                c_next = f_t * c_fore + i_t * T.tanh(T.dot(self.Wc, v) + self.bc)
                h_next = o_t * T.tanh(c_next)
                return h_next, c_next

            loc_for = T.zeros_like(self.seq_loc) + self.target_content_index
            al_for = self.a_for_left * T.exp(
                -self.b_for_left * T.abs_(
                    self.seq_loc[0:self.target_content_index] - loc_for[0:self.target_content_index]))
            am_for = self.a_for_middle * [1]
            a_for = T.concatenate([al_for, am_for])
            locate_for = T.zeros_like(self.seq_matrix[0:self.target_content_index + 1],
                                      dtype=T.config.floatX) + T.reshape(a_for, [-1, 1])
            loc_back = T.zeros_like(self.seq_loc) + self.target_content_index
            ar_back = self.a_back_right * T.exp(
                -self.b_back_right * T.abs_(
                    self.seq_loc[self.target_content_index + 1:] - loc_back[self.target_content_index + 1:]))
            ar_back = ar_back[::-1]
            a_back = T.concatenate([am_for, ar_back])
            locate_back = T.zeros_like(self.seq_matrix[self.target_content_index:], dtype=T.config.floatX) + T.reshape(
                a_back, [-1, 1])

            scan_result_forward, _forward = theano.scan(fn=encode_forward,
                                                        sequences=locate_for * X[0:self.target_content_index + 1],
                                                        outputs_info=[h, c])
            scan_result_backward, _backward = theano.scan(fn=encode_backward,
                                                          sequences=locate_back * X[self.target_content_index:][::-1],
                                                          outputs_info=[h, c])
            embedding_l = scan_result_forward[0]
            embedding_r = scan_result_backward[0]
            h_target_for = embedding_l[-1]
            h_target_back = embedding_r[-1]

            attention_h_target_l = embedding_l
            cont_l = T.concatenate([h_target_for, h_target_back])
            yuyi_l = T.transpose(cont_l)
            alpha_h_l = T.dot(T.dot(attention_h_target_l, self.alpha_h_W_L), yuyi_l)
            alpha_tmp_l = T.nnet.softmax(alpha_h_l)
            r_l = T.dot(alpha_tmp_l, embedding_l)
            h_star_L = T.tanh(T.dot(r_l, self.Wp_L))

            attention_h_target_r = embedding_r
            cont_r = T.concatenate([h_target_for, h_target_back])
            yuyi_r = T.transpose(cont_r)

            alpha_h_r = T.dot(T.dot(attention_h_target_r, self.alpha_h_W_R), yuyi_r)
            alpha_tmp_r = T.nnet.softmax(alpha_h_r)
            r_r = T.dot(alpha_tmp_r, embedding_r)
            h_star_R = T.tanh(T.dot(r_r, self.Wp_R))
            embedding = T.concatenate([h_star_L, h_star_R],
                                      axis=1)
            return embedding

        embedding = rnn(self.seq_matrix, self.tar_vector)
        embedding_for_train = embedding * self.srng.binomial(embedding.shape, p=0.5, n=1, dtype=embedding.dtype)
        embedding_for_test = embedding * 0.5

        self.pred_for_train = T.nnet.softmax(T.dot(embedding_for_train, self.Ws) + self.bs)
        self.pred_for_test = T.nnet.softmax(T.dot(embedding_for_test, self.Ws) + self.bs)

        self.l2 = sum([T.sum(param ** 2) for param in self.params]) - T.sum(self.Vw ** 2)
        self.loss_sen = -T.tensordot(self.solution, T.log(self.pred_for_train), axes=2)
        self.loss_l2 = 0.5 * self.l2 * self.regular
        self.loss = self.loss_sen + self.loss_l2

        grads = T.grad(self.loss, self.params)
        self.updates = collections.OrderedDict()
        self.grad = {}
        for param, grad in zip(self.params, grads):
            g = theano.shared(np.asarray(np.zeros_like(param.get_value()), \
                                         dtype=theano.config.floatX))
            self.grad[param] = g
            self.updates[g] = g + grad

        self.func_train = theano.function(
            inputs=[self.seq_idx, self.target, self.solution,
                    self.target_content_index, self.seq_loc, self.seq_len,
                    theano.In(h, value=self.h0),
                    theano.In(c, value=self.c0)],
            outputs=[self.loss, self.loss_sen, self.loss_l2],
            updates=self.updates,
            on_unused_input='warn')

        self.func_test = theano.function(
            inputs=[self.seq_idx, self.target, self.target_content_index, self.seq_loc, self.seq_len,
                    theano.In(h, value=self.h0),
                    theano.In(c, value=self.c0)],
            outputs=self.pred_for_test,
            on_unused_input='warn')

    def load_word_vector(self, fname, wordlist):
        loader = WordLoader()
        dic = loader.load_word_vector(fname, wordlist, self.dim_word)
        not_found = 0
        Vw = self.Vw.get_value()
        for word, index in wordlist.items():
            try:
                Vw[index] = dic[word]
            except:
                not_found += 1
        print 'not_found:', not_found
        self.Vw.set_value(Vw)
