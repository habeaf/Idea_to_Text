import torch
from torch import nn
import numpy as np


def xray(x):
    """
  Provide information about the type and structure of the input argument 'x'.

  Parameters:x: Any, the input argument to be inspected.

  Returns: None

  Prints information about the type and structure of 'x', including:
  - For lists: whether it contains tensors, length, and shape of the first element.
  - For numpy arrays or generics: shape information.
  - For dictionaries: length, shape of the key (interpreted as tensor), and shape of the value (interpreted as tensor).
  - For tuples: whether it contains tensors, length, and shape of the first element.
  - For torch tensors: shape information.
  """

    if isinstance(x, list):
        if isinstance(x[0], torch.Tensor):
            print(f"List of Tensors, length: {len(x)}, shape of the first element: {x[0].shape} ")
        else:
            print(f"List, length: {len(x)}, type of the first element: {type(x[0])}")

    elif isinstance(x, (np.ndarray, np.generic)):
        print("np.array, Shape:", x.shape)

    elif isinstance(x, dict):
        first_key = list(x.keys())[0]
        first_value = list(x.values())[0]
        if isinstance(first_value, torch.Tensor) and isinstance(first_key, torch.Tensor):
            print(f"Dict, Len: {len(x)}, Keys(Tensor): {first_key.shape}, Values(Tensor): {first_value.shape}")
        elif isinstance(first_key, torch.Tensor) and not isinstance(first_value, torch.Tensor):
            print(f"Dict, Len: {len(x)}, Keys(Tensor): {first_key.shape}, Values(Tensor): {type(first_value)}")
        elif isinstance(first_value, torch.Tensor) and not isinstance(first_value, torch.Tensor):
            print(f"Dict, Len: {len(x)}, Keys(Tensor): {type(first_key)}, Values(Tensor): {first_value.shape}")
        else:
            print("Dictionary with non-tensor values and first key and value types", type(first_key), type(first_value))

    elif isinstance(x, tuple):
        if isinstance(x[0], torch.Tensor):
            print(f"Tuple of Tensors, length: {len(x)}, shape of the first element: {x[0].shape}")
        else:
            print(f"Tuple, length: {len(x)}, type of the first element: {type(x[0])}")

    elif isinstance(x, torch.Tensor):
        print("Tensor, Shape:", x.shape)


def row_trim(x, lw, bg):
    """
    Trim rows from a numpy array based on given lower and upper bounds.

    Parameters:
    x (numpy.ndarray): Input array to be trimmed.
    lw (float): Lower bound threshold. Rows with elements below this threshold will be removed.
    bg (float): Upper bound threshold. Rows with elements above this threshold will be removed.

    Returns:
    numpy.ndarray: Trimmed array with rows removed based on the specified bounds.
    """
    marked_idxs = []
    shape = x.shape
    for i in range(shape[0]):
        for e in x[i]:
            if e > bg or e < lw:
                marked_idxs.append(i)
    z = np.delete(x, marked_idxs, axis=0)
    return z


def convert_tensor_to_list_of_tensors(x):
    """
    Convert a tensor into a list of tensors.

    Parameters:
    x (tensor): Input tensor to be converted.

    Returns:
    list: A list containing individual tensors extracted from the input tensor.
    """
    list_of_tensors = []

    for i in range(x.shape[0]):
        list_of_tensors.append(x[i, :])

    return list_of_tensors


def convert_list_of_tensors_to_tensor(x):
    """
    Convert a list of tensors into a single tensor by stacking them along a new dimension.

    Parameters:
    x (list): List of tensors to be converted.

    Returns:
    tensor: A single tensor obtained by stacking the tensors along a new dimension.
    """
    x = torch.stack(x)

    return x


def pmv(M):
    """
    Print the shape of each variable in a list.

    Parameters:
    M (list): List of variables.

    Returns:
    None
    """
    length = len(M)
    for i in range(length):
        print(M[i].shape)


class Lin_View(nn.Module):
    """
    A PyTorch module for performing a linear view operation on input tensors.

    Attributes:
    None
    """

    def __init__(self):
        """
        Initialize the Lin_View module.
        """
        super(Lin_View, self).__init__()

    def forward(self, x):
        """
        Perform the forward pass of the Lin_View module.

        Parameters:
        x (tensor): Input tensor.

        Returns:
        tensor: Modified tensor after applying the linear view operation.
        """
        return x.view(x.size()[0], -1)


def one_hot(y):
    """
    Convert numerical labels into one-hot encoded vectors.

    Parameters:
    y (ndarray): Input array containing numerical labels.

    Returns:
    ndarray: One-hot encoded vectors corresponding to the input labels.
    """
    y = y.reshape(len(y))
    n_values = np.max(y) + 1
    a = np.eye(n_values)
    print(a)
    print(a[0])
    print()
    return np.eye(n_values)[np.array(y, dtype=np.int32)]


def printt():
    print(" ")
    print(" ")
    print(" ")


def check_shape_and_type_of_loaderdata(loader):
    counter = 0
    sample = 0
    sample_target = 0
    for e, i in loader:
        if counter == 0:
            sample = e
            sample_target = i
            counter + counter + 1
        break

    print("________________________________________")

    print("For Loader data:")
    print("Shape data: ", e.shape, "Dtype data: ", e.dtype)
    print("Shape label: ", sample_target.shape, "      Dtype label: ", sample_target.dtype)
    print("________________________________________")

    return e, sample_target


def plot_signal(x, y, size_figure, title, type):
    if type == 0:

        fig = plt.figure(figsize=size_figure)
        fig.suptitle(title, fontweight="bold")
        fig.set_dpi = (500)
        plt.plot(x, y)
        plt.grid(axis='both', alpha=.3)
        plt.show()

    else:

        # plt.scatter(x, label)

        fig = plt.figure(figsize=size_figure)
        fig.suptitle(title, fontweight="bold")
        fig.set_dpi = (500)
        plt.scatter(x, y)
        plt.grid(axis='both', alpha=.3)
        plt.show()


def plot_signal_with_figtext(x, y, size_figure, title, type, figtext):
    # plt.figtext(0, 4, "T = 4K")
    if type == 0:

        fig = plt.figure(figsize=size_figure)
        fig.suptitle(title, fontweight="bold")
        fig.set_dpi = (500)
        plt.figtext(figtext[0], figtext[1], figtext[2])
        plt.grid(axis='both', alpha=.3)
        plt.plot(x, y)

        plt.show()

    else:

        # plt.scatter(x, label)

        fig = plt.figure(figsize=size_figure)
        fig.suptitle(title, fontweight="bold")
        fig.set_dpi = (500)
        plt.figtext(figtext[0], figtext[1], figtext[2])
        plt.scatter(x, y)
        plt.grid(axis='both', alpha=.3)
        plt.show()


def trim_max_values_in_np_array(feature_all):
    idx = np.argwhere(feature_all < -300)
    # a = np.array([1,2,3,4,5,6,7,8,9,10])
    # idx = np.stack([[2, 3], [7, 8]])
    feature_all[idx] = -300
    # np.delete(a, idx[:, 1:])
