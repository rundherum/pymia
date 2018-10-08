"""Enables the plotting of 3-dimensional visualizations.

Use `conda install -c clinicalgraphics vtk` to install the required Visualization Toolkit (VTK) with anaconda.

Examples:
    
    The first example computes the surface-to-surface distance of a unit sphere and unit cube.

    >>> # create sphere and cube
    >>> sphere = vtk.vtkSphereSource()
    >>> sphere.SetPhiResolution(100)
    >>> sphere.SetThetaResolution(100)
    >>> cube = vtk.vtkCubeSource()
    >>> 
    >>> distance_filter = vtk.vtkDistancePolyDataFilter()
    >>> distance_filter.SetInputConnection(0, sphere.GetOutputPort())
    >>> distance_filter.SetInputConnection(1, cube.GetOutputPort())
    >>> distance_filter.Update()
    >>> distance = distance_filter.GetOutput()
    
    The second example computes the surface-to-surface distance of a prediction to a ground truth image.

    >>> image1, transform1 = read_and_transform_image('PREDICTION.mha')
    >>> image2, transform2 = read_and_transform_image('GROUND-TRUTH.mha')
    >>> isocontour1 = compute_isocontour(image1)
    >>> isocontour2 = compute_isocontour(image2)
    >>> model = compute_surface_to_surface_distance(isocontour1, isocontour2)

    
"""
import pathlib

import SimpleITK as sitk
import vtk


def compute_isocontour(image_label: vtk.vtkImageData, label: int=1) -> vtk.vtkPolyData:
    """Calculates the isocontour from a label image.

    Args:
        image_label (vtk.vtkImageData): The label image.
        label (int): The label.

    Returns:
        (vtk.vtkPolyData): The extracted isocontour.
    """
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(image_label)
    marchingCubes.SetValue(0, label)
    marchingCubes.Update()

    return marchingCubes.GetOutput()


def compute_surface_to_surface_distance(isocontour1: vtk.vtkPolyData, isocontour2: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Calculates the surface-to-surface distance between two isocontours.

    Args:
        isocontour1 (vtk.vtkPolyData): The first isocontour.
        isocontour2 (vtk.vtkPolyData): The second isocontour.

    Returns:
        (vtk.vtkPolyData)
    """
    cleaner1 = vtk.vtkCleanPolyData()
    cleaner1.SetInputData(isocontour1)

    cleaner2 = vtk.vtkCleanPolyData()
    cleaner2.SetInputData(isocontour2)

    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetInputConnection(0, cleaner1.GetOutputPort())
    distance_filter.SetInputConnection(1, cleaner2.GetOutputPort())
    distance_filter.Update()

    return distance_filter.GetOutput()


def read_and_transform_image(file_path: str) -> (vtk.vtkImageData, vtk.vtkTransform):
    """Reads an image as VTK image.

    VTK images have no direction matrix, therefore we apply a transformation to the image after loading to have in the
    correct space. Note that only meta and NIfTI images are supported.

    Args:
        file_path (str): The file path.

    Returns:
        (vtk.vtkImageData): The image.

    Raises:
        ValueError:
    """
    # check file extension to use correct reader
    extension = ''.join(pathlib.Path(file_path).suffixes)

    if extension == '.mha' or extension == '.mdh':
        reader = vtk.vtkMetaImageReader()
    elif extension == '.nii' or extension == '.nii.gz':
        reader = vtk.vtkNIFTIImageReader()
    else:
        raise ValueError('Unknown image format {}'.format(extension))

    reader.SetFileName(file_path)
    reader.Update()
    image = reader.GetOutput()

    # VTK images have no direction matrix, therefore load the same image with itk and apply a transformation
    image_sitk = sitk.ReadImage(file_path)
    direction = image_sitk.GetDirection()
    origin = image_sitk.GetOrigin()

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    matrix.SetElement(0, 0, direction[0])
    matrix.SetElement(0, 1, direction[3])
    matrix.SetElement(0, 2, direction[6])
    matrix.SetElement(1, 0, direction[1])
    matrix.SetElement(1, 1, direction[4])
    matrix.SetElement(1, 2, direction[7])
    matrix.SetElement(2, 0, direction[2])
    matrix.SetElement(2, 1, direction[5])
    matrix.SetElement(2, 2, direction[8])

    transform = vtk.vtkTransform()
    transform.Identity()
    # transform.Update()
    # transform.PostMultiply()
    # transform.Translate(-origin[0], -origin[1], -origin[2])
    transform.Translate(-origin[0], -origin[1], -origin[2])
    transform.Concatenate(matrix)

    return image, transform
