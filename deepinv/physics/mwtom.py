import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from deepinv.physics.forward import Physics
from pylops import LinearOperator

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
            nshapes=3, device=torch.device("cpu"), **kwargs, ):
        super().__init__(**kwargs)

    def A(self, x):
        return

    def A_dagger(self, y):
        print()
        return


class stepBOp(LinearOperator):
    def __init__(self, EFgen, GS, x_domain, y_domain):
        self.EFgen = EFgen
        self.GS = GS
        self.x_domain = x_domain
        self.y_domain = y_domain
        super().__init__()

    def _matvec(self, x):
        # forward linear operator
        _, self.ET = self.EFgen.generate_total_electric_field(x, self.x_domain, self.y_domain, full_pixel=True)
        b = x.squeeze()
        b = torch.flatten(torch.transpose(b, -2, -1), -2)
        b = b.unsqueeze(-1)
        aux = torch.mul(self.ET, b)
        y = torch.matmul(-1j * self.GS, aux)
        return y.reshape(-1)

    def _rmatvec(self, y):
        # adjoint linear operator
        B = torch.matmul(1j * self.GS.H, y.reshape(self.GS.shape[0], -1))
        C = torch.mul(self.ET.conj(), B)
        C = torch.sum(C, axis=1)
        return C


class MWTstepB(Physics):
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
            receiver_radius=3,
            transmitter_radius=3,
            wave_type=0,
            device=torch.device("cpu"),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.electric_field_generator = ElectricFieldGenerator(
            no_of_pixels=img_width,
            no_of_receivers=no_of_receivers,
            no_of_transmitters=no_of_transmitters,
            max_diameter=max_diameter,
            wavelength=wavelength,
            receiver_radius=receiver_radius,
            transmitter_radius=transmitter_radius,
            wave_type=wave_type
            )

        image_domain = np.linspace(-max_diameter, max_diameter, img_width)
        self.x_domain, self.y_domain = np.meshgrid(image_domain, -image_domain)
        self.GS = self.electric_field_generator.compute_GS(self.x_domain, self.y_domain)

        self.stepBLinOp = stepBOp(self.electric_field_generator, self.GS, self.x_domain, self.y_domain)

    def A(self, x):
        return self.stepBLinOp._matvec(x)

    def A_adjoint(self, y):
        return self.stepBLinOp._rmatvec(y)

        return
