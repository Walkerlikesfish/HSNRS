"""
This script include the class making a loopy graph, and applying Belief propagation through the graph

v0.0:
4 neighbours only
basic propagation simultaneously -> matrix

v0.1:
4 neighbours only
basic propagation in priority -> matrix


Yu Liu @ ETRO VUB
2017
"""

import numpy as np
import matplotlib.image
import matplotlib.pyplot

# define default directions dictionary
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DLOSS = 4


class XBP:
    def __init__(self, mat_smooth, mat_base_belief, mat_weight):
        # init the basic parameter by the input
        self.num_class = mat_smooth.shape[0]
        self.height = mat_base_belief.shape[0]
        self.width = mat_base_belief.shape[1]
        t_num_class = mat_base_belief.shape[2]
        if t_num_class != self.num_class:
            raise Exception("Class number does not coherent !")

        # mat_msg used to store the status of messages in the process
        # [x,x,0,x] reserved for data_loss
        self.mat_msg = np.ones(
            shape=(self.height, self.width, 5, self.num_class),
            dtype=np.float32
        )
        # fill in the mat_msg
        # [normalise] by the number of class
        self.mat_msg[:, :, DLOSS, :] = mat_base_belief * (
            1 / np.sum(mat_base_belief, axis=2, keepdims=True))

        # mat_smooth for smooth function
        self.mat_smooth = np.ones(
            shape=(self.num_class, self.num_class),
            dtype=np.float32
        )
        # fill in the mat_smooth
        # [normalise] Do not forget to normalise
        self.mat_smooth = mat_smooth * (
            1 / np.sum(mat_smooth, axis=1, keepdims=True))

        self.mat_weight = mat_weight
        # self.mat_diff = mat_diff

        # working array
        self._mat_working = np.ones(
            shape=(self.height, self.width, 4, self.num_class, self.num_class),
            dtype=np.float32)
        self._mat_working /= self.num_class

        # belief storage arrays
        self._mat_beliefprod = np.ndarray(
            shape=(self.height, self.width, self.num_class),
            dtype=np.float32)
        self._belief = np.ndarray(
            shape=(self.height, self.width),
            dtype=np.int)

        # temp sum array
        self._mat_t_sum = np.ones(
            shape=(self.height, self.width),
            dtype=np.float32
        )

    def x_pass_msg_four(self):
        '''
        Helper class to iterate through four neighbours
        :return:
        '''
        for i_dir in xrange(4):
            self.x_pass_msg_one(i_dir)

    def x_pass_msg_one(self, i_dir):
        '''
        Pass message in i_dir for once
        :param i_dir: the direction defined in the header
        :return:
        '''
        # Define the working matrix
        # mat_working: is where i->j the i is
        #   the idea is to calculate in mat_working with all combination(l, l')
        #   then do sum against all the (l')
        #   save to the dir_to [!Normalisation!]
        if i_dir == UP:
            mat_working = self._mat_working[1:, :, UP]
            mat_source = self.mat_msg[1:, :]
            mat_source_w = self.mat_weight[1:, :]
            mat_dst = self.mat_msg[:-1, :]
            mat_storage = self._mat_t_sum[:-1, :]
            dir_from = DOWN
            dir_to = UP
            # mat_diff_c = self.mat_diff[1:, :, dir_to]
        elif i_dir == RIGHT:
            mat_working = self._mat_working[:, :-1, RIGHT]
            mat_source = self.mat_msg[:, :-1]
            mat_source_w = self.mat_weight[:, :-1]
            mat_dst = self.mat_msg[:, 1:]
            mat_storage = self._mat_t_sum[:, 1:]
            dir_from = LEFT
            dir_to = RIGHT
            # mat_diff_c = self.mat_diff[:, :-1, dir_to]
        elif i_dir == DOWN:
            mat_working = self._mat_working[:-1, :, DOWN]
            mat_source = self.mat_msg[:-1, :]
            mat_source_w = self.mat_weight[:-1, :]
            mat_dst = self.mat_msg[1:, :]
            mat_storage = self._mat_t_sum[1:, :]
            dir_from = UP
            dir_to = DOWN
            # mat_diff_c = self.mat_diff[:-1, :, dir_to]
        elif i_dir == LEFT:
            mat_working = self._mat_working[:, 1:, LEFT]
            mat_source = self.mat_msg[:, 1:]
            mat_source_w = self.mat_weight[:, 1:]
            mat_dst = self.mat_msg[:, :-1]
            mat_storage = self._mat_t_sum[:, :-1]
            dir_from = RIGHT
            dir_to = LEFT
            # mat_diff_c = self.mat_diff[:, 1:, dir_to]

        # initial the working mat by copying the mat_smooth to every (i,j)
        # smooth(l, l')
        mat_working[:, :] = self.mat_smooth

        # smooth(l, l') * data_loss(l', y_i) <- msg(i,i)
        t_mat_dloss = mat_source[:, :, DLOSS, np.newaxis, :]
        mat_working[:, :] *= t_mat_dloss

        # smooth(l, l') * data_loss(l', y_i) * PI(k){ (msg_k_i/j(l')) }
        for t_id in (UP, RIGHT, DOWN, LEFT):
            # don't include the message from the pixel we're sending to
            if t_id == dir_to:
                continue
            # but do include the other three
            mat_working[:] *= mat_source[:, :, t_id, np.newaxis, :]

        # sum l' in L
        # TODO: here adding a mask to implement the priority
        np.sum(mat_working, axis=3, out=mat_dst[:, :, dir_from])

        # calculate the sum of the labels
        # TO [!normalise!]
        t_mat = mat_dst[:, :, dir_from]
        np.sum(t_mat, axis=2, out=mat_storage)
        # in case of normalisation overflow
        mask_zero = mat_storage == 0
        mat_storage[mask_zero] = 1
        np.reciprocal(mat_storage, out=mat_storage)
        # updating the message by multiply
        mat_dst[:, :, dir_from] *= mat_storage[:, :, np.newaxis]
        # weighing the dst
        mat_dst[:, :, dir_from] *= mat_source_w[:, :, np.newaxis]
        # weighting the dst by the color similarity
        # np.reciprocal(mat_diff_c, out=mat_diff_c)
        # mat_dst[:, :, dir_from] *= mat_diff_c[:, :, np.newaxis]


    def x_calc_belief(self):
        # b_i(l) = PI(k){msg_mat(k->i)}
        self.mat_msg.prod(axis=2, out=self._mat_beliefprod)
        self._mat_beliefprod.argmax(axis=2, out=self._belief)
        return self._belief

    def xh_show_msg(self):
        pass