import numpy as np
import torch
from mwtomography.dataloader.image.image_generator import ImageGenerator

try:
    import mwtomography
except:
    odl = ImportError("The mwtomography package is not installed.")
    
def random_MWTphantom(image_generator, x_domain, y_domain, nshapes=3):
    """
    Generate a random  phantom.
    """
    if isinstance(mwtomography, ImportError):
        raise ImportError(
            "mwtomography is needed to use generate random phantoms. "
        ) from mwtomography

    if nshapes == 'random':
        no_of_shapes = int(np.ceil((3 * np.random.uniform()) + 1e-2))
        shapes = image_generator.shape_generator.generate_shapes(no_of_shapes)
    elif type(nshapes) == int:
        no_of_shapes = nshapes
        shapes = image_generator.shape_generator.generate_shapes(no_of_shapes)
    elif nshapes == 'fixed_pattern':
        shapes = image_generator.shape_generator.generate_shapes_pattern()

    image = Image()
    image.generate_relative_permittivities(x_domain, y_domain, shapes)
    return image


class RandomMWTPhantomDataset(torch.utils.data.Dataset):
    """
    Dataset of images with shapes (circles or rectangles) randomly located. The phantoms are generated on the fly.
    The phantoms are generated using the mwtomography library (https://github.com/ccaiafa/mwtomography.git).

    :param class image_generator: ImageGenerator class defined in mwtomography library
    :param int/str nshapes: number of shapes or 'fixed_pattern' or 'random'
    :param transform: Transformation to apply to the output image.
    :param float length: Length of the dataset. Useful for iterating the data-loader for a certain nb of iterations.
    """

    def __init__(self, image_generator, nshapes, transform=None, length=np.inf):
        image_domain = np.linspace(-image_generator.max_diameter, image_generator.max_diameter, image_generator.no_of_pixels)
        self.x_domain, self.y_domain = np.meshgrid(image_domain, -image_domain)
        self.transform = transform
        self.n_data = image_generator.no_of_images
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        :return tuple : A tuple (phantom, 0) where phantom is a torch tensor of shape (n_data, size, size).
        """
        phantom_np = np.array([random_phantom(self.space) for i in range(self.n_data)])
        phantom = torch.from_numpy(phantom_np).float()
        if self.transform is not None:
            phantom = self.transform(phantom)
        return phantom, 0