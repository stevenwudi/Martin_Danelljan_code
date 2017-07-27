"""
Di Wu's re-implemenation of the paper
[1] Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg and Joost van de Weijer.
    "Adaptive Color Attributes for Real-Time Visual Tracking".
    Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.

email: stevenwudi@gmail.com  2017/08/01
"""
import numpy as np
import h5py
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

        if os.path.isfile(w2c_file_path):
            f = h5py.File(w2c_file_path)
            self.w2c = f['w2crs'].value.transpose(1,0)
        else:
            print("W2C file not found!")

        self.name = 'cvpr_2014_color_name'

    def train(self, im, init_rect):
        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        self.res.append(init_rect)
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.im_sz = im.shape[:2]
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.output_sigma = np.sqrt(np.prod(self.target_sz)) * self.output_sigma_factor

        grid_y = np.arange(np.floor(self.target_sz[0]) - np.floor(self.target_sz[0] / 2))
        grid_x = np.arange(np.floor(self.target_sz[1]) - np.floor(self.target_sz[0] / 2))
        rs, cs = np.meshgrid(grid_x, grid_y)
        self.y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = self.fft2(self.y)
        # store pre-computed cosine window
        self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # extract the feature map of the local image patch to train the classifer
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        self.get_features()

    def detect(self):
        pass

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

    def get_features(self):

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        if self.non_compressed_features == 'gray':
            self.out_npca = rgb2gray(self.im_crop)/255. - 0.5

        if self.compressed_features == 'cn':
            self.out_pca = self.im2c(self.im_crop)

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
        # 'F' ,eams to flatten in column-major
        # Because we convert w2c from matlab which is a column-major programming language, Duh >:<
        index_im = np.floor(np.ndarray.flatten(RR, 'F') / 8.) + \
                   32 * np.floor(np.ndarray.flatten(GG, 'F') / 8.) + \
                   32 * 32 * np.floor(np.ndarray.flatten(BB, 'F') / 8.)

        out = np.reshape(self.w2c[index_im.astype('int')], (im.shape[0], im.shape[1], self.w2c.shape[1]), 'F')
        return out