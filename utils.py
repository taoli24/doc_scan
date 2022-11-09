from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from typing import Any


def checkPath(path_: str, *paths, ext: str = '', errors: str = 'ignore') -> tuple:
    """
    Join the paths together, adds an extension if not already included
    in path, then checks if path exists.

    :param path_: Main file path, should be a str
    :param paths: Remaining file paths, should be a tuple[str]
    :param ext: File extension, should be a str
    :param errors: Whether to 'ignore', 'warn' or 'raise' errors, should be str
    :return: path_, exist - tuple[str, bool]
    """
    if errors not in ["ignore", "warn", "raise"]:
        raise ValueError("The parameter errors must be either 'ignore', 'warn' or 'raise'")

    path_ = joinPath(path_, *paths, ext=ext)

    exist = True if os.path.exists(path_) else False

    if not exist and errors != 'ignore':
        msg = f"No such file or directory: '{path_}'"
        logging.warning(msg)
        if errors == 'warn':
            warnings.warn(msg)
        elif errors == 'raise':
            raise FileNotFoundError(msg)
    return path_, exist


def joinPath(path_: str, *paths, ext: str = '') -> str:
    """
    Join the paths together, adds an extension if not already included
    in path.

    :param path_: Main file path, should be a str
    :param paths: Remaining file paths, should be a tuple[str]
    :param ext: File extension, should be a str
    :return: path_ - str
    """
    path_, path_ext = os.path.splitext(os.path.join(path_, *paths))
    if ext and path_ext != ext:
        path_ext = (ext if ext and '.' in ext else f'.{ext}')
    return path_ + path_ext


def makePath(path_: str, *paths, errors: str = 'ignore') -> str:
    """
    Check if the path exists and creates the path when required.

    :param path_: Main file path, should be a str
    :param paths: Remaining file paths, should be a tuple[str]
    :param errors: Whether to 'ignore', 'warn' or 'raise' errors, should be str
    :return: path_ - str
    """
    path_, exist = checkPath(path_, *paths, errors=errors)
    if not exist:
        os.makedirs(path_)
        logging.info(f"Path has been made: '{path_}'")
    return path_


def listPath(path_: str, *paths, ext: list | tuple | str = '', return_file_path: bool = False,
             errors: str = 'raise') -> tuple:
    """
    Join the paths together, return list of files within directory or
    specific files by extension.

    :param path_: Main file path, should be a str
    :param paths: Remaining file paths, should be a tuple[str]
    :param ext: File extension, should be a list | tuple | str
    :param return_file_path: Whether to return file name or file path, should be a bool
    :param errors: Whether to 'ignore', 'warn' or 'raise' errors, should be str
    :return: path_, files - tuple[str, list[str]]
    """
    if errors not in ["ignore", "warn", "raise"]:
        raise ValueError("The parameter errors must be either 'ignore', 'warn' or 'raise'")

    if isinstance(ext, str):
        ext = [ext]

    path_, exist = checkPath(path_, *paths, errors=errors)

    files = []
    for file in os.listdir(path_):
        if os.path.splitext(file)[1].replace('.', '') in ext:
            files.append(joinPath(path_, file) if return_file_path else file)
    return path_, files


def sepPath(path_: str, direction: str = 'lr', max_split: int = 1) -> tuple:
    """
    Separate the path into left and right by direction and split size.

    :param path_: Path to a folder or file, should be a str
    :param direction: Whether to split the path from 'lr' or 'rl', should be str
    :param max_split: The splitting size, should be an int
    :return: left, right - tuple[str, str]
    """
    sep_path = path_.split('\\')
    if max_split not in range(1, len(sep_path)):
        raise ValueError(f"'max_split' must be within range of 1 and directory depth, got: {max_split}")

    if direction == 'lr':
        left, right = sep_path[:max_split], sep_path[max_split:]
    elif direction == 'rl':
        left, right = sep_path[:(len(sep_path) - max_split)], sep_path[(len(sep_path) - max_split):]
    else:
        raise ValueError("The parameter direction must be either 'lr' or 'rl'")
    return '\\'.join(left), '\\'.join(right)


def getLastPath(path_: str, include_ext: bool = True) -> str:
    """
    Separates the last path from the given path and includes the option
    to remove file extension.

    :param path_: Path to a folder or file, should be a str
    :param include_ext: Whether to include file extension, should be a bool
    :return: last_path - str
    """
    last_path = sepPath(path_, direction='rl', max_split=1)[0]
    if not include_ext:
        return os.path.splitext(last_path)[0]
    return last_path


def update(obj: object, kwargs: dict) -> object:
    """
    Update the objects attributes, if given attributes are present
    in object and match existing data types.

    :param obj: The object that is being updated, should be an object
    :param kwargs: Keywords and values to be updated, should be a dict
    :return: obj - object
    """
    for key, value in kwargs.items():
        if not hasattr(obj, key):
            raise AttributeError(f"'{obj.__class__.__name__}' object has no attribute '{key}'")
        else:
            attr_ = getattr(obj, key)
            if isinstance(attr_, (type(value), type(None))) or value is None:
                setattr(obj, key, value)
            else:
                raise TypeError(f"'{key}': Expected type '{type(attr_).__name__}', got '{type(value).__name__}'")
    return obj


def load(dir_: str, name: str, ext: str = '', errors: str = 'raise') -> Any:
    """
    Load the data with appropriate method. Pickle will deserialise the
    contents of the file and json will load the contents.

    :param dir_: Directory of file, should be a str
    :param name: Name of file, should be a str
    :param ext: File extension, should be a str
    :param errors: Whether to 'ignore', 'warn' or 'raise' errors, should be str
    :return: data - Any
    """
    if errors not in ["ignore", "warn", "raise"]:
        raise ValueError("The parameter errors must be either 'ignore', 'warn' or 'raise'")

    if not ext:
        ext = os.path.splitext(name)[1]

    if not ext:
        msg = f"The parameters 'name' or 'ext' must include file extension, got: '{name}', '{ext}'"
        logging.warning(msg)
        if errors == 'warn':
            warnings.warn(f"Name '{name}' must include file extension")
        elif errors == 'raise':
            raise ValueError(msg)
        return

    path_, _ = checkPath(dir_, name, ext=ext, errors=errors)

    if ext == '.json':
        with open(path_, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif ext == '.txt':
        with open(path_, 'r') as file:
            data = file.read()
    else:
        with open(path_, 'rb') as file:
            data = pickle.load(file)
    logging.info(f"File '{name}' data was loaded")
    return data


def save(dir_: str, name: str, data: Any, indent: int = 4, errors: str = 'raise') -> bool:
    """
    Save the data with appropriate method. Pickle will serialise the
    object, while json will dump the data with indenting to allow users
    to edit and easily view the encoded data.

    :param dir_: Directory of file, should be a str
    :param name: Name of file, should be a str
    :param data: Data to be saved, should be an Any
    :param indent: Data's indentation within the file, should be an int
    :param errors: If 'ignore', suppress errors, should be str
    :return: completed - bool
    """
    if errors not in ["ignore", "warn", "raise"]:
        raise ValueError("The parameter errors must be either 'ignore', 'warn' or 'raise'")

    path_ = joinPath(dir_, name)
    ext = os.path.splitext(name)[1]

    checkPath(dir_, errors=errors)
    if not os.path.exists(dir_):
        msg = f"No such file or directory: '{dir_}'"
        logging.warning(msg)
        if errors == 'warn':
            warnings.warn(msg)
        elif errors == 'raise':
            raise FileNotFoundError(msg)
        return False

    if not ext:
        logging.warning(f"File '{name}' must include file extension in name")
        if errors == 'raise':
            warnings.warn(f"File '{name}' must include file extension in name")
        return False

    if ext == '.json':
        with open(path_, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=indent)
    elif ext == '.txt':
        with open(path_, 'w') as file:
            file.write(str(data))
    elif isinstance(data, object):
        with open(path_, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
    else:
        msg = f"Saving method was not determined, failed to save file, got: {type(data)}"
        logging.warning(msg)
        if errors == 'warn':
            warnings.warn(msg)
        elif errors == 'raise':
            raise FileNotFoundError(msg)
        return False
    logging.info(f"File '{name}' was saved")
    return True
