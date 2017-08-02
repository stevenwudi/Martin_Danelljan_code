# coding=utf-8
"""
Di Wu's re-implemenation of the Fast Discriminative Scale Space Tracker (fDSST) [1],
which is an extension of the VOT2014 winning DSST tracker [2].
The code provided by [3] is used for computing the HOG features.


[1]	Martin Danelljan, Gustav Fahad Khan, Michael Felsberg.
	Discriminative Scale Space Tracking.
	Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

[2] Martin Danelljan, Gustav Fahad Shahbaz Khan and Michael Felsberg.
    "Accurate Scale Estimation for Robust Visual Tracking".
    Proceedings of the British Machine Vision Conference (BMVC), 2014.

[3] Piotr Doller
    "Piotr抯 Image and Video Matlab Toolbox (PMT)."
    http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html.

Contact:
Di Wu
http://stevenwudi.github.io/
email: stevenwudi@gmail.com  2017/08/01
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scripts.pyhog import pyhog
import cv2

class bmvc_2014_pami_2014_fDSST:
    def __init__(self,
                 padding=2.0,
                 output_sigma_factor=float(1/16.),
                 scale_sigma_factor=float(1/16.),
                 lambda_value=1e-2,
                 interp_factor=0.025,
                 num_compressed_dim=18,
                 refinement_iterations=1,
                 translation_model_max_area=np.inf,
                 interpolate_response=1,
                 resize_factor=1,
                 number_of_scales=17,
                 number_of_interp_scales=33.,
                 scale_model_factor=1.0,
                 scale_step=1.02,
                 scale_model_max_area=512,
                 s_num_compressed_dim='MAX',
                 featureRatio=4.0,
                 visualisation=1,
                 ):
        """

        :param padding: extra area surrounding the target
        :param output_sigma_factor: standard deviation for the desired translation filter output
        :param scale_sigma_factor: standard deviation for the desired scale filter output
        :param lambda_value: regularisation weight (denoted "lambda" in the paper)
        :param interp_factor: tracking model learning rate (denoted "eta" in the paper)
        :param num_compressed_dim: the dimensionality of the compressed features
        :param refinement_iterations: number of iterations used to refine the resulting position in a frame
        :param translation_model_max_area: maximum area of the translation model
        :param interpolate_response: interpolation method for the translation scores
        :param resize_factor: initial resize
        :param number_of_scales: number of scale levels
        :param number_of_interp_scales: number of scale levels after interpolation
        :param scale_model_factor: relative size of the scale sample
        :param scale_step: scale increment factor (denoted 'a" in the paper)
        :param scale_model_max_area: the maximum size of scale examples
        :param s_num_compressed_dim: number of compressed scale feature dimensions
        :param featureRatio: HOG window size
        :param visualisation: flag for visualistion
        """
        self.padding = padding
        self.output_sigma_factor = output_sigma_factor
        self.scale_sigma_factor = scale_sigma_factor
        self.lambda_value = lambda_value
        self.interp_factor = interp_factor
        self.num_compressed_dim = num_compressed_dim
        self.refinement_iterations = refinement_iterations
        self.translation_model_max_area = translation_model_max_area
        self.interpolate_response = interpolate_response
        self.resize_factor = resize_factor
        self.number_of_scales = number_of_scales
        self.number_of_interp_scales = number_of_interp_scales
        self.scale_model_factor = scale_model_factor
        self.scale_step = scale_step
        self.scale_model_max_area = scale_model_max_area
        self.s_num_compressed_dim = s_num_compressed_dim
        self.visualisation = visualisation
        self.featureRatio = featureRatio

        if self.number_of_scales > 0:
            self.scale_sigma = self.number_of_interp_scales * self.scale_sigma_factor
            self.scale_exp = (np.arange(self.number_of_scales) - np.floor(self.number_of_scales/2)) * \
                             (self.number_of_interp_scales/self.number_of_scales)
            self.interp_scale_exp = np.arange(self.number_of_interp_scales) - np.floor(self.number_of_interp_scales/2)

            self.scaleSizeFactors = self.scale_step ** self.scale_exp
            self.interpScaleFactors = self.scale_step ** self.interp_scale_exp

            self.ys = np.exp(-0.5 * self.scale_exp**2 / self.scale_sigma**2)
            self.ysf = np.fft.fft(self.ys)
            self.scale_wnidow = np.hanning(self.ysf.shape[0])

        self.name = 'bmvc_2014_pami_2014_fDSST'

    def train(self, im, init_rect):
        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        self.res.append(init_rect)
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.init_target_sz = self.target_sz
        self.im_sz = im.shape[:2]

        if np.prod(self.target_sz.shape) > self.translation_model_max_area:
            self.currentScaleFactor = np.sqrt(np.prod(self.init_target_sz) / self.translation_model_max_area)
        else:
            self.currentScaleFactor = 1.0
        # target size at the initial scale
        self.base_target_sz = self.target_sz / self.currentScaleFactor
        # window size, taking padding into account
        self.patch_size = np.floor(self.base_target_sz * (1 + self.padding))
        self.output_sigma = np.sqrt(np.prod(np.floor(self.base_target_sz / self.featureRatio))) * self.output_sigma_factor
        self.use_sz = np.floor(self.patch_size/self.featureRatio)

        grid_y = np.arange(np.floor(self.use_sz[0])) - np.floor(self.use_sz[0] / 2)
        grid_x = np.arange(np.floor(self.use_sz[1])) - np.floor(self.use_sz[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        self.y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = self.fft2(self.y)

        if self.number_of_scales > 0:
            # make sure the scale model is not too large so as to save computation time
            if self.scale_model_factor**2 * np.prod(self.init_target_sz) > self.scale_model_max_area:
                self.scale_model_factor = np.sqrt(self.scale_model_max_area/np.prod(self.init_target_sz))

            # set the scale model size
            self.scale_model_sz = np.floor(self.init_target_sz * self.scale_model_factor)
            # force reasonable scale changes
            self.min_scale_factor = self.scale_step ** np.ceil(np.log(np.max(5. / self.patch_size)) / np.log(self.scale_step))
            self.max_scale_factor = self.scale_step ** np.floor(np.log(np.min(np.array([im.shape[0], im.shape[1]]) * 1.0 / self.base_target_sz)) / np.log(self.scale_step))

            #if self.s_num_compressed_dim == 'MAX':

        # TODO: Compute coefficients for the tranlsation filter
        # extract the feature map of the local image patch to train the classifer
        self.im_crop_origin = self.get_subwindow(im, self.pos, self.patch_size*self.currentScaleFactor)
        # redudant below, but for the sake of formality
        #self.im_crop = imresize(self.im_crop_origin, self.patch_size)
        self.im_crop = self.im_crop_origin
        # initiliase the appearance
        self.z_npca, self.z_pca = self.get_features(self.im_crop)

        # TODO: Compute coefficents for the scale filter


    def detect(self, im, frame):
        # TODO: write detection function
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

    def get_features(self, im_crop):
        """
        :param im_crop:
        :return:
        """
        # because the hog output is (dim/4)-2>1:
        if self.patch_size.min() < 12:
            scale_up_factor = 12. / np.min(im_crop)
            im_patch_resized = imresize(im_crop, np.asarray(self.patch_size * scale_up_factor).astype('int'))

        features_hog = pyhog.features_pedro(im_crop.astype(np.float64) / 255.0, 4)
        cell_gray = self.cell_gray(self.im_crop)

        temp_pca = np.concatenate([features_hog, cell_gray[:, :, None]], axis=2)
        out_pca = temp_pca.reshape([temp_pca.shape[0]*temp_pca.shape[1], temp_pca.shape[2]])

        return [], out_pca

    def cell_gray(self, img):
        """
        Average the intensity over a single hog-cell
        :param img:
        :return:
        """
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        if len(img.shape) == 3:
            gray_img = rgb2gray(img)
        else:
            gray_img = img

        # compute the integral image
        integral_img = cv2.integral(gray_img)
        cell_size = int(self.featureRatio)

        i1 = np.array(range(cell_size, gray_img.shape[0]+1, cell_size))
        i2 = np.array(range(cell_size, gray_img.shape[1]+1, cell_size))
        A1, A2 = np.meshgrid(i1 - cell_size, i2 - cell_size)
        B1, B2 = np.meshgrid(i1, i2 - cell_size)
        C1, C2 = np.meshgrid(i1 - cell_size, i2)
        D1, D2 = np.meshgrid(i1, i2)
        # cell_sum = integral_img[A1, A2] - integral_img(B1, i2 - cell_size) -\
        #            integral_img(i1 - cell_size) + integral_img(i1 - cell_size, + i2 - cell_size)

        cell_sum = integral_img[A1, A2] - integral_img[B1, B2] - integral_img[C1, C2] + integral_img[D1, D2]
        cell_gray = cell_sum.T / (cell_size**2 * 255) - 0.5
        return cell_gray