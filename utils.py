import functools
import inspect
import logging
import os
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml
from IPython.display import display
from PIL import Image


class CheckPoint:
    def __init__(self, filename):
        self._time = 0.0
        self._iter = 0
        self.filename: str = filename
        self.load()
        pass

    def __repr__(self):
        return f"Checkpoint([iter={self.iter}\ntime={time.time() - self.time}])"
        pass

    def __str__(self):
        return self.__repr__()

    @property
    def time(self):
        return self._time

    @property
    def iter(self):
        return self._iter

    def load(self):
        try:
            checkpoint = self._read_checkpoints()
            pass
        except FileNotFoundError:
            checkpoint = {'iter': 0, 'time': time.time()}
            pass

        self._iter = int(checkpoint['iter'] if 'iter' in checkpoint else self.iter)
        self._time = float(checkpoint['time'] if 'time' in checkpoint else self.time)
        return self

    def save(self, gen_iter=0, time=0.0):
        self._iter = gen_iter if gen_iter > 0 else self.iter
        self._time = time if time > 0.0 and not self.iter == 0 else self.time
        self._write_checkpoints()
        return self

    def _write_checkpoints(self):
        with open(self.filename, 'w') as f:
            f.write(f"iter={self.iter}\ntime={self.time}")
            pass
        pass

    def _read_checkpoints(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Checkpoint file `{self.filename}` doesn't exist")

        checkpoint = {}
        with open(self.filename, 'r') as f:
            for line in f:
                name, var = line.partition("=")[::2]
                checkpoint[name.strip()] = float(var)
                pass
            pass
        return checkpoint

    pass


class DummyLogger(logging.Logger):
    """
    Dummy class to disable logging from env variable
    """

    def __init__(self, name="dummy"):
        super(DummyLogger, self).__init__(name)
        pass

    def log(self, level, msg, *args, **kwargs) -> None:
        pass

    def info(self, msg, *args, **kwargs) -> None:
        pass

    def debug(self, msg, *args, **kwargs) -> None:
        pass

    def warning(self, msg, *args, **kwargs) -> None:
        pass

    def warn(self, msg, *args, **kwargs) -> None:
        pass

    def error(self, msg, *args, **kwargs):
        return super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return super().critical(msg, *args, **kwargs)

    pass


def makedirs(pathnames: list, parents=True, exist_ok=True):
    for pathname in pathnames:
        Path(pathname).mkdir(parents=parents, exist_ok=exist_ok)
    pass


def _save_img_path_maker(save_path=None, default_path="./"):
    spath = save_path if save_path is not None else default_path
    makedirs([spath])
    return spath


def _squeeze_fn(img, path):
    return np.squeeze(np.array([path([img[i:i + 1]]) for i in range(img.shape[0])]))


# from https://stackoverflow.com/a/40301488
def deprecated(reason):
    if isinstance(reason, (type(b''), type(u''))):
        def decorator(func1):
            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."
                pass

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator
    elif inspect.isclass(reason) or inspect.isfunction(reason):
        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2
    else:
        raise TypeError(repr(type(reason)))
    pass


# noinspection PyUnresolvedReferences
def get_image_paths(directory):
    return [
        x.path for x in os.scandir(directory)
        if x.name.endswith(".jpg") or x.name.endswith(".png")
    ]


def load_images(image_paths, convert=None):
    iter_all_images = (cv2.resize(cv2.imread(fn), (256, 256)) for fn in image_paths)
    all_images = []
    if convert:
        iter_all_images = (convert(img) for img in iter_all_images)
        pass
    for i, image in enumerate(iter_all_images):
        if i == 0:
            all_images = np.empty((len(image_paths),) + image.shape, dtype=image.dtype)
            pass
        all_images[i] = image
        pass
    return all_images


def get_transpose_axes(n):
    if n % 2 == 0:
        y_axes = list(range(1, n - 1, 2))
        x_axes = list(range(0, n - 1, 2))
        pass
    else:
        y_axes = list(range(0, n - 1, 2))
        x_axes = list(range(1, n - 1, 2))
        pass
    return y_axes, x_axes, [n - 1]


def stack_images(images):
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
    ).reshape(new_shape)


def save_image(test_src, test_dst, path_src, path_dst, batch_size,
               im_save_path, is_mask=False, filename="img"):
    def _get_squeeze(t, p):
        return (_squeeze_fn(t, p) * 2) - 1 if is_mask else _squeeze_fn(t, p)

    save_path = _save_img_path_maker(im_save_path, "./")
    save_path = f"{save_path}/{filename}"

    figure1 = np.stack([
        test_src,
        _get_squeeze(test_src, path_src),
        _get_squeeze(test_src, path_dst),
    ], axis=1)
    figure2 = np.stack([
        test_dst,
        _get_squeeze(test_dst, path_dst),
        _get_squeeze(test_dst, path_src),
    ], axis=1)

    figure = np.concatenate([figure1, figure2], axis=0)
    figure = figure.reshape((4, batch_size // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(figure)
    if im_save_path is not None:
        image.save(save_path)
        pass
    display(image)
    pass


def save_loss_data(save_path, d):
    spath = _save_img_path_maker(save_path, "./losses")
    with open(f"{spath}/loss.txt", "a") as f:
        print(d)
        f.write(d)
        pass
    pass


def show_debug_window(image: np.ndarray, name="debug", width=600, height=600):
    cv2.namedWindow(name)
    cv2.resizeWindow(name, width, height)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(name)
    pass


def save_debug_img(image: np.ndarray, filename="debug.png"):
    if len(filename.split('.')) == 1:
        filename += ".png"
        pass
    try:
        cv2.imwrite(filename, image)
    except IOError as e:
        print("Couldn't save image " + filename)
        pass
    pass


@deprecated("Deprecated in favor of save_image")
def showG(test_src, test_dst, path_src, path_dst, batch_size):
    figure1 = np.stack([
        test_src,
        np.squeeze(np.array([path_src([test_src[i:i + 1]]) for i in range(test_src.shape[0])])),
        np.squeeze(np.array([path_dst([test_src[i:i + 1]]) for i in range(test_src.shape[0])])),
    ], axis=1)
    figure2 = np.stack([
        test_dst,
        np.squeeze(np.array([path_dst([test_dst[i:i + 1]]) for i in range(test_dst.shape[0])])),
        np.squeeze(np.array([path_src([test_dst[i:i + 1]]) for i in range(test_dst.shape[0])])),
    ], axis=1)

    figure = np.concatenate([figure1, figure2], axis=0)
    figure = figure.reshape((4, batch_size // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(figure))
    pass


@deprecated("Deprecated in favor of save_image")
def showG_mask(test1, test2, path1, path2, batch_size):
    figure1 = np.stack([
        test1,
        ((np.squeeze(np.array([path1([test1[i:i + 1]]) for i in range(test1.shape[0])]))) * 2) - 1,
        ((np.squeeze(np.array([path2([test1[i:i + 1]]) for i in range(test1.shape[0])]))) * 2) - 1,
    ], axis=1)
    figure2 = np.stack([
        test2,
        ((np.squeeze(np.array([path2([test2[i:i + 1]]) for i in range(test2.shape[0])]))) * 2) - 1,
        ((np.squeeze(np.array([path1([test2[i:i + 1]]) for i in range(test2.shape[0])]))) * 2) - 1,
    ], axis=1)

    figure = np.concatenate([figure1, figure2], axis=0)
    figure = figure.reshape((4, batch_size // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(figure))
    pass


def showG_eyes(test1, test2, bm_eyes1, bm_eyes2, batch_size):
    figure1 = np.stack([
        (test1 + 1) / 2,
        bm_eyes1,
        bm_eyes2 * (test1 + 1) / 2,
    ], axis=1)
    figure2 = np.stack([
        (test2 + 1) / 2,
        bm_eyes2,
        bm_eyes2 * (test2 + 1) / 2,
    ], axis=1)

    figure = np.concatenate([figure1, figure2], axis=0)
    figure = figure.reshape((4, batch_size // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip(figure * 255, 0, 255).astype('uint8')
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)

    display(Image.fromarray(figure))
    pass


# noinspection DuplicatedCode
def save_preview_image(test1, test2,
                       path_a, path_b,
                       path_bgr_a, path_bgr_b,
                       path_mask_a, path_mask_b,
                       batch_size, save_fn="preview.jpg"):
    figure_a = np.stack([
        test1,
        np.squeeze(np.array([path_bgr_b([test1[i:i + 1]]) for i in range(test1.shape[0])])),
        (np.squeeze(np.array([path_mask_b([test1[i:i + 1]]) for i in range(test1.shape[0])]))) * 2 - 1,
        np.squeeze(np.array([path_b([test1[i:i + 1]]) for i in range(test1.shape[0])])),
    ], axis=1)
    figure_b = np.stack([
        test2,
        np.squeeze(np.array([path_bgr_a([test2[i:i + 1]]) for i in range(test2.shape[0])])),
        (np.squeeze(np.array([path_mask_a([test2[i:i + 1]]) for i in range(test2.shape[0])]))) * 2 - 1,
        np.squeeze(np.array([path_a([test2[i:i + 1]]) for i in range(test2.shape[0])])),
    ], axis=1)

    figure = np.concatenate([figure_a, figure_b], axis=0)
    figure = figure.reshape((4, batch_size // 2) + figure.shape[1:])
    figure = stack_images(figure)
    figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
    cv2.imwrite(save_fn, figure)
    pass


def load_yaml(path_configs):
    with open(path_configs, 'r') as f:
        return yaml.load(f)
    pass


def show_loss_config(loss_config):
    """
    Print out loss configuration. Called in loss function automation.

    Argument:
        loss_config: A dictionary. Configuration regarding the optimization.
    """
    for config, value in loss_config.items():
        print(f"{config} = {value}")
        pass
    pass
