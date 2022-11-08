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

    :param path_: main file path, should be a str
    :param paths: remaining file paths, should be a tuple[str]
    :param ext: file extension, should be a str
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
    path_ = os.path.join(path_, *paths)
    path_ext = os.path.splitext(path_)[1]
    if ext and path_ext != ext:
        path_ = path_[:-len(path_ext)] + (ext if ext and '.' in ext else f'.{ext}')
    return path_


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
    :param ext: file extension, should be a list | tuple | str
    :param return_file_path: whether to return file name or file path, should be a bool
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
    :param ext: file extension, should be a str
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
