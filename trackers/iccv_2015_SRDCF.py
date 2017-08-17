"""
Di Wu's re-implemenation of the Fast Discriminative Scale Space Tracker (fDSST) [1],
which is an extension of the VOT2014 winning DSST tracker [2].
The code provided by [3] is used for computing the HOG features.

[1]	Martin Danelljan, Gustav Fahad Khan, Michael Felsberg.
Learning Spatially Regularized Correlation Filters for Visual Tracking.
In Proceedings of the International Conference in Computer Vision (ICCV), 2015.

Contact:
Di Wu
http://stevenwudi.github.io/
email: stevenwudi@gmail.com  2017/08/016
"""

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.misc import imresize


class iccv_2015_SRDCF:
    def __init__(self,
                 search_area_scale=4,
                 output_sigma_factor=0.0625,
                 lambda_value=1e-2,
                 learning_rate=0.025,
                 refinement_iterations=1,
                 search_area_shape='square',
                 filter_max_area=2500,
                 number_of_scales=7,
                 scale_step=1.01,
                 interpolate_response=4,
                 num_GS_iter=4,
                 cell_size=4,
                 cell_selection_thresh=0.75**2,
                 use_reg_window=True,
                 reg_window_power=2,
                 reg_window_edge=3,
                 reg_window_min=0.1,
                 reg_sparsity_threshold=0.05,

                 ):
        """
        Initialisation function
        :param search_area_scale: size of the training/detection area proportional to the target size
        :param output_sigma_factor:
        :param lambda_value:
        :param learning_rate:
        :param refinement_iterations:
        :param filter_max_area:
        :param number_of_scales:
        :param scale_step:
        :param interplate_response: correlation score interpolation strategy:
                                    0 - off, 1 - feature grid, 2 - pixel grid, 4 - Newton's method
        :param num_GS_iter:
        :param cell_selection_thresh: Threshold for reducing the cell size in low-resolution cases
        :param use_reg_window: flag of using windowed reglurisation
        :param reg_window_edge: the impact of the spatial regularization (value at the target border),
                                depends on the detection size and the feature dimensionality
        :param reg_window_min: the minimum value of the regularization window
        """
        self.search_area_scale = search_area_scale
        self.output_sigma_factor = output_sigma_factor
        self.interpolate_response = interpolate_response
        self.filter_max_area = filter_max_area
        self.search_area_shape = search_area_shape
        self.cell_size = cell_size
        self.feature_ratio = float(self.cell_size)
        self.cell_selection_thresh = cell_selection_thresh

        # regularisation parameters
        self.use_reg_window = use_reg_window
        self.reg_window_power = reg_window_power
        self.reg_window_edge = reg_window_edge
        self.reg_window_min = reg_window_min
        self.reg_sparsity_threshold = reg_sparsity_threshold

        self.name = 'iccv_2015_SRDCF'

    def train(self, im, init_rect):
        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        self.res.append(init_rect)
        self.target_sz = np.asarray(init_rect[2:])
        self.init_target_sz = self.target_sz[::-1]

        search_area = np.prod(self.target_sz / self.feature_ratio * self.search_area_scale)
        # initiate some cell size for trainng purposes
        self. cell_size, self.search_area, self.current_scale_factor, self.sz, self.use_sz = \
            self.resize_cell_size(search_area, self.cell_selection_thresh, self.filter_max_area, self.feature_ratio,
                                  self.init_target_sz, self.search_area_scale, self.cell_size, self.search_area_shape)
        # consturct the label funnction
        self.y, self.yf, self.yf_vec, self.interp_sz, self.support_sz, self.cos_window = self.construct_translational_label_function(
            self.init_target_sz, self.feature_ratio, self.output_sigma_factor, self.use_sz, self.interpolate_response)

        # compute the indices for the real, positive and negative parts of the spectrum
        dft_sym_ind, dft_pos_ind, dft_neg_ind, dfs_matrix = self.partition_spectrum2(self.use_sz)

        # regularisation parameters
        self.construct_regularisation_window(self.init_target_sz, self.feature_ratio, self.use_sz,
                                             self.reg_window_edge, self.reg_window_min, self.reg_window_power,
                                             self.reg_sparsity_threshold)


    def resize_cell_size(self, search_area, cell_selection_thresh, filter_max_area, feature_ratio,
                  init_target_sz, search_area_scale, cell_size, search_area_shape):
        """
        Choosing the cell size according to the predefined maximum search area,
        :param search_area:
        :param cell_selection_thresh:
        :param filter_max_area:
        :param feature_ratio:
        :param init_target_sz:
        :param search_area_scale:
        :return:
        """
        if search_area < cell_selection_thresh * filter_max_area:
            cell_size = int(np.min(feature_ratio, np.max([1, np.ceil(np.sqrt(np.prod(
                init_target_sz * search_area_scale)/(cell_selection_thresh*filter_max_area)))])))
            feature_ratio = cell_size
            search_area = np.prod(init_target_sz / feature_ratio * search_area_scale)

        if search_area > filter_max_area:
            current_scale_factor = np.sqrt(search_area / filter_max_area)
        else:
            current_scale_factor = 1.0

        # target size at the initial scale
        base_target_size = init_target_sz / current_scale_factor

        if search_area_shape == 'proportional':
            sz = np.floor(base_target_size * search_area_scale)
        elif search_area_shape == 'square':
            sz = np.tile(np.sqrt(np.prod(base_target_size * search_area_scale)), 2)

        sz = np.round(sz /feature_ratio) * feature_ratio
        use_sz = np.floor(sz/feature_ratio).astype('int')

        return cell_size, search_area, current_scale_factor, sz, use_sz

    def construct_translational_label_function(self, init_target_sz, feature_ratio,
                                               output_sigma_factor, use_sz, interpolate_response):

        output_sigma = np.sqrt(np.prod(np.floor(init_target_sz/feature_ratio))) * output_sigma_factor
        grid_y = np.roll(np.arange(np.floor(use_sz[0])) - np.floor(use_sz[0] / 2), int(-np.floor(use_sz[0]/2)))
        grid_x = np.roll(np.arange(np.floor(use_sz[1])) - np.floor(use_sz[1] / 2), int(-np.floor(use_sz[1]/2)))
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        yf = self.fft2(y)
        # create vectorised desired correlation output
        yf_vec = yf.flatten()

        if interpolate_response == 1:
            interp_sz = use_sz * feature_ratio
        else:
            interp_sz = use_sz

        # the search area size
        support_sz = np.prod(use_sz)
        # store pre-computed cosine window
        cos_window = np.outer(np.hanning(use_sz[0]), np.hanning(use_sz[1]))

        return y, yf, yf_vec, interp_sz, support_sz, cos_window

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        return np.fft.fft2(x, axes=(0, 1))

    def partition_spectrum2(self, dft_sz):
        """
        partitions the spectrum of a 2-dimensional signal with dimensions dft_sz into the real part,
         a set of "positive" frequenceis and the corresponding "negative" frequencies
        :return:
        """
        #construct the index vector for the half of the spectrum to be saved
        spec_dim = np.ceil((dft_sz.astype('float')+1)/2).astype('int')
        dim_even = np.mod(dft_sz, 2) == 0

        if dim_even[1]:
            # linear indices of the part of the spectrum that is needed [g_0, g_+]
            dft_ind = np.concatenate([range(spec_dim[0]), range(dft_sz[0],(spec_dim[1]-1)*dft_sz[0]+spec_dim[0])])
        # linear indices of the part of the spectrum that have a symmetric counterpart g_+
            dft_pos_ind = np.concatenate([range(1, spec_dim[0]-1),  range(dft_sz[0], (spec_dim[1]-1)*dft_sz[0]),
                                          (spec_dim[1] - 1) * dft_sz[0] + range(1, spec_dim[0]-1)])

        else:
            dft_ind = np.concatenate([range(spec_dim[0]), range(dft_sz[0], spec_dim[1]*dft_sz[0])])
            dft_pos_ind = np.concatenate([range(1, spec_dim[0]), range(dft_sz[0], spec_dim[1]*dft_sz[0])])

        # linear indices for the part of the spectrum that is real, g_0
        dft_sym_ind = np.setdiff1d(dft_ind, dft_pos_ind)

        # construct the indices for the corresponding negative frequencies
        pos_dft_id = np.zeros(dft_sz)
        pos_dft_id[np.unravel_index(dft_pos_ind, dft_sz, order='F')] = np.array(range(len(dft_pos_ind)))+1
        neg_dft_loc = self.reflect_spectrum_2d(pos_dft_id)
        corresponding_dft_pos_id = np.nonzero(neg_dft_loc)
        dft_neg_ind_unsorted = neg_dft_loc[corresponding_dft_pos_id]
        dft_neg_ind_order = np.argsort(dft_neg_ind_unsorted)
        dft_neg_ind = np.ravel_multi_index(corresponding_dft_pos_id, dft_sz, order='F')[dft_neg_ind_order]

        # the discrete fourier series output indices
        dfs_sym_ind = np.array(range(len(dft_sym_ind)))
        dfs_real_ind = dfs_sym_ind[-1] + 1 + 2 * np.array(range(len(dft_pos_ind)))
        dfs_imag_ind = dfs_sym_ind[-1] + 2 + 2 * np.array(range(len(dft_pos_ind)))

        # construct the transofmration matrix from dft to dtf 9the real fourier series)
        dfs_matrix = self.dft2dfs_matrix(dft_sym_ind, dft_pos_ind, dft_neg_ind, dfs_sym_ind, dfs_real_ind, dfs_imag_ind)

        # all dft_sym_ind, dft_pos_ind, dft_neg_ind starts from index of 0 but 'F' (row indexed)
        return dft_sym_ind, dft_pos_ind, dft_neg_ind, dfs_matrix

    def reflect_spectrum_2d(self, pos_dft_id):
        return np.roll(np.roll(np.flip(np.flip(pos_dft_id, 0), 1), shift=1, axis=0), shift=1, axis=1)

    def dft2dfs_matrix(self, dft_sym_ind, dft_pos_ind, dft_neg_ind, dfs_sym_ind, dfs_real_ind, dfs_imag_ind):
        """
        Construct a sparse matrix that transforms the discrete foueir transform (DFT) to the
        real disrete fourier series (DFS), given the input and ouput index permutations.
        :param dft_sym_ind:
        :param dft_pos_ind:
        :param dft_neg_ind:
        :param dfs_sym_ind:
        :param dfs_real_ind:
        :param dfs_imag_ind:
        :return:
        """
        i_sym = dfs_sym_ind
        j_sym = dft_sym_ind
        v_sym = np.ones(len(dft_sym_ind))

        i_real_pos = dfs_real_ind
        j_real_pos = dft_pos_ind
        v_real_pos = 1 / np.sqrt(2) * np.ones(len(dft_pos_ind))

        i_real_neg = dfs_real_ind
        j_real_neg = dft_neg_ind
        v_real_neg = 1 / np.sqrt(2) * np.ones(len(dft_neg_ind))

        i_imag_pos = dfs_imag_ind
        j_imag_pos = dft_pos_ind
        v_imag_pos = 1 / (1j * np.sqrt(2)) * np.ones(len(dft_pos_ind))

        i_imag_neg = dfs_imag_ind
        j_imag_neg = dft_neg_ind
        v_imag_neg = -1 / (1j * np.sqrt(2)) * np.ones(len(dft_neg_ind))

        i_tot = np.concatenate([i_sym, i_real_pos, i_real_neg, i_imag_pos, i_imag_neg])
        j_tot = np.concatenate([j_sym, j_real_pos, j_real_neg, j_imag_pos, j_imag_neg])
        v_tot = np.concatenate([v_sym, v_real_pos, v_real_neg, v_imag_pos, v_imag_neg])

        dft_length = len(dft_sym_ind) + len(dft_pos_ind) + len(dft_neg_ind)

        dfs_matrix = scipy.sparse.bsr_matrix((v_tot, [i_tot, j_tot]), shape=(dft_length, dft_length))

        return dfs_matrix

    def construct_regularisation_window(self, init_target_sz, feature_ratio, use_sz,
                                        reg_window_edge, reg_window_min, reg_window_power,
                                        reg_sparsity_threshold):
        reg_scale = 0.5 * init_target_sz / feature_ratio

        wrg = np.arange(np.floor(use_sz[0])) - np.floor(use_sz[0] / 2)
        wcg = np.arange(np.floor(use_sz[1])) - np.floor(use_sz[1] / 2)
        wrs, wcs = np.meshgrid(wrg, wcg)
        # construct the regularisation window
        reg_window = (reg_window_edge - reg_window_min) * (np.abs(wrs/reg_scale[0])**reg_window_power +
                                                           np.abs(wcs/reg_scale[1])**reg_window_power) + reg_window_min

        # compute the DFT and enforce sparsity
        reg_window_dft = self.fft2(reg_window) /np.prod(use_sz)
        reg_window_dft_sep = np.stack((np.real(reg_window_dft), np.imag(reg_window_dft)), axis=2)
        reg_window_dft_sep[np.abs(reg_window_dft_sep) < reg_sparsity_threshold * np.abs(reg_window_dft_sep.flatten()).max()] = 0
        reg_window_dft = reg_window_dft_sep[:, :, 0] + 1j * reg_window_dft_sep[:, :, 1]

        # do the inverse transoform,  correct window minimum
        reg_window_sparse = np.real(np.fft.ifft2(reg_window_dft))
        reg_window_dft[0, 0] = reg_window_dft[0, 0] - np.prod(use_sz) * reg_window_sparse.min() + reg_window_min


        return

    def cconvmtx2(self, reg_window_dft):
        """
        construct the regularisation matrix
        :param reg_window_dft:
        :return:
        """
        nrow, ncol = reg_window_dft.shape
        num_elem = nrow * ncol

        H1 = scipy.sparse.spalloc(num_elem, nrow, nrow * len(np.nonzero(reg_window_dft)[0]))

        # create the first n columns

