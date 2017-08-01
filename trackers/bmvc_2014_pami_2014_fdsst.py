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
                 number_of_interp_scales=33,
                 scale_model_factor=1.0,
                 scale_step=1.02,
                 scale_model_max_area=512,
                 s_num_compressed_dim='MAX',
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

    def train(self, im, init_rect):
        # TODO: write training function
        pass

    def detect(self, im, frame):
        # TODO: write detection function
        pass