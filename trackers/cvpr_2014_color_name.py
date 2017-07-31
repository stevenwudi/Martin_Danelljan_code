"""
Di Wu's re-implemenation of the paper
[1] Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg and Joost van de Weijer.
    "Adaptive Color Attributes for Real-Time Visual Tracking".
    Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

email: stevenwudi@gmail.com  2017/08/01
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os


class cvpr_2014_color_name:
    def __init__(self,
                padding=1.0,
                output_sigma_factor=float(1/16.),
                sigma=0.2,
                lambda_value=1e-2,
                learning_rate=0.075,
                compression_learning_rate=0.15,
                non_compressed_features='gray',
                compressed_features='cn',
                num_compressed_dim=2,
                visualization=1,
                w2c_file_path='trackers/w2crs.mat'
                ):
        """
        :param padding: extra area surrounding the target
        :param output_sigma_factor: spatial bandwidth (proportional to target)
        :param sigma: gaussian kernel bandwidth
        :param lambda_value: regularization (denoted "lambda" in the paper)
        :param learning_rate: learning rate for appearance model update scheme
                            (denoted "gamma" in the paper)
        :param compression_learning_rate: learning rate for the adaptive dimensionality reduction
                            (denoted "mu" in the paper)
        :param non_compressed_features: features that are not compressed, a cell with strings
                            (possible choices: 'gray', 'cn')
        :param compressed_features: features that are compressed, a cell with strings
                            (possible choices: 'gray', 'cn')
        :param num_compressed_dim: the dimensionality of the compressed features
        :param visualization: flag for visualisation
        :param w2c_file_path: color naming file path
        :return:
        """
        self.padding = padding
        self.output_sigma_factor = output_sigma_factor
        self.sigma = sigma
        self.lambda_value = lambda_value
        self.learning_rate = learning_rate
        self.compression_learning_rate = compression_learning_rate
        self.non_compressed_features = non_compressed_features
        self.compressed_features = compressed_features
        self.num_compressed_dim = num_compressed_dim
        self.visualization = visualization
        self.currentScaleFactor = 1  # no scale method used here
        self.feature_type = 'cvpr_2014_color_name'

        if os.path.isfile(w2c_file_path):
            f = h5py.File(w2c_file_path)
            self.w2c = f['w2crs'].value.transpose(1,0)
        else:
            print("W2C file not found!")

        self.name = 'cvpr_2014_color_name'

    def train(self, im, init_rect):
        """
        train the appearance model from the frist frame
        """
        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        self.res.append(init_rect)
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.im_sz = im.shape[:2]
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.output_sigma = np.sqrt(np.prod(self.target_sz)) * self.output_sigma_factor

        grid_y = np.arange(np.floor(self.patch_size[0])) - np.floor(self.patch_size[0] / 2)
        grid_x = np.arange(np.floor(self.patch_size[1])) - np.floor(self.patch_size[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        self.y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = self.fft2(self.y)
        # store pre-computed cosine window
        self.cos_window = np.outer(np.hanning(self.patch_size[0]), np.hanning(self.patch_size[1]))

        # extract the feature map of the local image patch to train the classifer
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        # initiliase the appearance
        self.z_npca, self.z_pca = self.get_features(self.im_crop)

        # if dimensionality reduction is used: update the projection matrix
        if self.compressed_features:
            self.projection_matrix = self.calculate_projection(self.z_pca, old_cov_matrix=False)

        # project the features of the new appearance example using the new projection matrix
        self.zp = self.feature_projection(self.z_npca, self.z_pca, self.projection_matrix, self.cos_window)

        # calculate the new classifier coefficients
        self.kf = self.fft2(self.dense_gauss_kernel(self.sigma, self.zp))
        self.alpha_num = np.multiply(self.yf, self.kf)
        self.alpha_den = np.multiply(self.kf, (self.kf + self.lambda_value))

    def detect(self, im, frame):
        """
        :param im: new dectection image
        :param frame: frame number
        :return: position of the new frame
        """
        # compute the compressed learnt appearance
        self.zp = self.feature_projection(self.z_npca, self.z_pca, self.projection_matrix, self.cos_window)

        # extract the feature map of the local image patch
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        self.xo_npca, self.xo_pca = self.get_features(self.im_crop)

        # do the dimensionality reduction and windowing
        self.x = self.feature_projection(self.xo_npca, self.xo_pca, self.projection_matrix, self.cos_window)

        # calculate the response of the classifier
        self.kf = self.fft2(self.dense_gauss_kernel(self.sigma, self.x, self.zp))
        self.response = np.real(np.fft.ifft2(np.divide(np.multiply(self.alpha_num, self.kf), self.alpha_den)))

        # target location is at the maximum response
        v_centre, h_centre = np.unravel_index(self.response.argmax(), self.response.shape)
        self.vert_delta, self.horiz_delta = [v_centre - np.floor(self.response.shape[0] / 2),
                                             h_centre - np.floor(self.response.shape[1] / 2)]
        self.pos = np.array(self.pos) + np.array([self.vert_delta, self.horiz_delta])

        # extract the feature map of the local image path to train the classifier
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        self.xo_npca, self.xo_pca = self.get_features(self.im_crop)
        #  update the appearance
        self.z_npca = (1 - self.learning_rate) * self.z_npca + self.learning_rate * self.xo_npca
        self.z_pca = (1 - self.learning_rate) * self.z_pca + self.learning_rate * self.xo_pca

        if self.compressed_features:
            self.projection_matrix = self.calculate_projection(self.z_pca, old_cov_matrix=True)

        # project the features of the new appearance example using the new projection matrix
        self.x = self.feature_projection(self.xo_npca, self.xo_pca, self.projection_matrix, self.cos_window)

        # calculate the new classifier coefficients
        kf_new = self.fft2(self.dense_gauss_kernel(self.sigma, self.x))
        alpha_num_new = np.multiply(self.yf, kf_new)
        alpha_den_new = np.multiply(kf_new, (kf_new + self.lambda_value))
        # subsequence frames, update the model
        self.alpha_num = (1 - self.learning_rate) * self.alpha_num + self.learning_rate * alpha_num_new
        self.alpha_den = (1 - self.learning_rate) * self.alpha_den + self.learning_rate * alpha_den_new

        # we also require the bounding box to be within the image boundary
        # self.res.append([min(self.im_sz[1] - self.target_sz[1], max(0, self.pos[1] - self.target_sz[1] / 2.)),
        #                  min(self.im_sz[0] - self.target_sz[0], max(0, self.pos[0] - self.target_sz[0] / 2.)),
        #                  self.target_sz[1], self.target_sz[0]])
        self.res.append([self.pos[1] - self.target_sz[1]/2., self.pos[0] - self.target_sz[0]/2.,
                         self.target_sz[1], self.target_sz[0]])
        return self.pos

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        return np.fft.fft2(x, axes=(0, 1))

    def get_subwindow(self, im, pos, sz):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """

        if np.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0] / 2)
        xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1] / 2)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= self.im_sz[0]] = self.im_sz[0] - 1

        xs[xs < 0] = 0
        xs[xs >= self.im_sz[1]] = self.im_sz[1] - 1

        return im[np.ix_(ys, xs)]

    def get_features(self, im_crop):
        xo_npca, xo_pca = None, None
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        if self.non_compressed_features == 'gray':
            xo_npca = rgb2gray(im_crop)/255. - 0.5

        if self.compressed_features == 'cn':
            xo_pca_temp = self.im2c(im_crop)
            xo_pca = np.reshape(xo_pca_temp,
                                (np.prod([xo_pca_temp.shape[0], xo_pca_temp.shape[1]]), xo_pca_temp.shape[2]))
            # # 'F' , means to flatten in column-major
            # xo_pca = np.reshape(xo_pca_temp,
            #                     (np.prod([xo_pca_temp.shape[0], xo_pca_temp.shape[1]]), xo_pca_temp.shape[2]),
            #                     order='F')

        return xo_npca, xo_pca

    def im2c(self, im):
        """
        Calcalate Color Names according to the paper:
        [3] J. van de Weijer, C. Schmid, J. J. Verbeek, and D. Larlus.
        "Learning color names for real-world applications."
        TIP, 2009
        :param im:
        :return:
        """
        RR = im[:, :, 0]
        GG = im[:, :, 1]
        BB = im[:, :, 2]

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
        # 'F' , means to flatten in column-major
        # Because we convert w2c from matlab which is a column-major programming language, Duh >:<
        index_im = np.floor(np.ndarray.flatten(RR, 'F') / 8.) + \
                   32 * np.floor(np.ndarray.flatten(GG, 'F') / 8.) + \
                   32 * 32 * np.floor(np.ndarray.flatten(BB, 'F') / 8.)

        out = np.reshape(self.w2c[index_im.astype('int')], (im.shape[0], im.shape[1], self.w2c.shape[1]), 'F')
        return out

    def feature_projection(self, xo_npca, xo_pca, projection_matrix, cos_window):
        """
        Calcaultes the compressed feature map by mapping the PCA features with the projection matrix
        and concatinates this with the non-PCA features. The feature map is than multiplied with a cosine-window.
        :return:
        """
        if not self.compressed_features:
            z = xo_npca
        else:
            # project the PCA-features using the projection matrix and reshape to a window
            x_proj_pca = np.dot(xo_pca, projection_matrix).reshape(
                (cos_window.shape[0], cos_window.shape[1], projection_matrix.shape[1]))

            # concatinate the feature windows
            if not self.non_compressed_features:
                z = x_proj_pca
            else:
                # gray scale concatenate with color names
                if len(xo_npca.shape) != len(x_proj_pca.shape):
                    z = np.concatenate((xo_npca[:, :, None], x_proj_pca), axis=2)

        features = np.multiply(z, cos_window[:, :, None])
        return features

    def dense_gauss_kernel(self, sigma, x, y=None):
        """
        Gaussian Kernel with dense sampling.
        Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
        between input images X and Y, which must both be MxN. They must also
        be periodic (ie., pre-processed with a cosine window). The result is
        an MxN map of responses.

        If X and Y are the same, ommit the third parameter to re-use some
        values, which is faster.
        :param sigma: feature bandwidth sigma
        :param x:
        :param y: if y is None, then we calculate the auto-correlation
        :return:
        """
        N = np.prod(x.shape)
        xf = self.fft2(x)
        xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

        if y is None:
            # auto-correlation of x
            yf = xf
            yy = xx
        else:
            yf = self.fft2(y)
            yy = np.dot(y.flatten().transpose(), y.flatten())  # squared norm of y

        xyf = np.multiply(xf, np.conj(yf))
        xy = np.real(np.fft.ifft2(np.sum(xyf, axis=2)))

        k = np.exp(-1. / sigma**2 * np.maximum(0, (xx + yy - 2 * xy)) / N)

        return k

    def calculate_projection(self, z_pca, old_cov_matrix=False):
        # compute the mean appearance
        data_mean = np.mean(z_pca, axis=0)
        # substract the mean from the appearance to get the data matrix
        data_matrix = np.subtract(z_pca, data_mean[None, :])
        # calculate the covariance matrix
        cov_matrix = np.cov(data_matrix.T)
        # calculate the principal components (pca_basis) and corresponding variances
        if old_cov_matrix:
            cov_matrix = (1 - self.compression_learning_rate) * self.old_cov_matrix + \
                         self.compression_learning_rate * cov_matrix
        else:
            cov_matrix = cov_matrix
        U, s, V = np.linalg.svd(cov_matrix)
        S = np.diag(s)

        # calculate the projection matrix as the first principal components
        # and extract their corresponding variances
        projection_matrix = U[:, :self.num_compressed_dim]
        projection_variances = S[:self.num_compressed_dim, :self.num_compressed_dim]
        # initialise the old covariance matrix using the computed projection matrix and variance
        if old_cov_matrix:
            self.old_cov_matrix = (1 - self.compression_learning_rate) * self.old_cov_matrix \
                                  + self.compression_learning_rate * \
                                    np.dot(np.dot(projection_matrix, projection_variances), projection_matrix.T)
        else:
            self.old_cov_matrix = np.dot(np.dot(projection_matrix, projection_variances), projection_matrix.T)

        return projection_matrix


