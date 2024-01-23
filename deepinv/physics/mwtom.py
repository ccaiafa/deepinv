import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from deepinv.physics.forward import Physics

from mwtomography.dataloader.electric_field.electric_field_generator import ElectricFieldGenerator


class MWTomography(Physics):
    r"""
    (MicroWave) Tomography nonlinear operator.

    It computes the two steps:
    (1) Calculate the total electric field Et such that Et = Ei + GD*LAMBDA*Et (linear problem);
    (2) Calculate measurements Es = GS*LAMBDA*Et,
    where GD and GS are the Green matrices and LAMBDA is a diagonal matrix containing in it main diagonal the vectorized complex object image.

    :param int img_width: width/height of the square image input.
    :param float wavelength: working wavelength
    :param int no_of_receivers: number of receiver antennas
    :param int no_of_transmitters: number of transmitter antennas
    :param float max_diameter: maximum object diameter
    :param float min_permittivity: minimum object permittivity
    :param float max_permittivity: maximum object permittivity
    :param float receiver_radius: receiver distance to center
    :param float transmitter_radius: transmitter distance to center
    :param bool wave_type: 0=linear, 1:planar
    :param str shape: "circle", "rectangle"
    :param int nshapes: number of shapes in the image
    :param str device: gpu or cpu.
    """

    def __init__(
        self,
        img_width,
        wavelength,
        no_of_receivers,
        no_of_transmitters,
        max_diameter=1,
        min_permittivity=1.0001,
        max_permittivity=1.5,
        receiver_radius=3,
        transmitter_radius=3,
        wave_type=0,
        shape="circle",
        nshapes=3, device=torch.device("cpu"), **kwargs,):

        super().__init__(**kwargs)

        self.T = 1
        self.noise_model =  2

    def A(self, x):
        print()
        return
    
    def A_dagger(self, y):
        print()
        return

