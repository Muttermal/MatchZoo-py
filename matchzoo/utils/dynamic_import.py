# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26 14:19
# @Author  : zhangguangyi
# @File    : dynamic_import.py

import importlib
import importlib.util
import logging
import sys
logger = logging.getLogger(__name__)


def import_module(module_name, package=None):
    """An approximate implementation of import."""
    absolute_name = importlib.util.resolve_name(module_name, package)
    try:
        return sys.modules[absolute_name]
    except KeyError:
        pass

    path = None
    if '.' in absolute_name:
        parent_name, _, child_name = absolute_name.rpartition('.')
        parent_module = import_module(parent_name)
        path = parent_module.__spec__.submodule_search_locations
    for finder in sys.meta_path:
        spec = finder.find_spec(absolute_name, path)
        if spec is not None:
            break
    else:
        msg = f'No module named {absolute_name!r}'
        raise ModuleNotFoundError(msg, name=absolute_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[absolute_name] = module
    spec.loader.exec_module(module)
    if path is not None:
        setattr(parent_module, child_name, module)
    return module


def import_class(class_name, module_name, package="matchzoo"):
    module = import_module(module_name, package)
    try:
        my_class = getattr(module, class_name)
        return my_class
    except Exception as e:
        logger.info(e)
        raise ImportError(f"找不到{class_name}模块")


if __name__ == "__main__":
    test_module = import_module(".models", "matchzoo")
    test_class = import_class("Bert", ".models", "matchzoo")
    test_optim = import_class("AdamW", ".optimization", "transformers")

    test_schedule = import_module(".optim", "torch")
    test_ = import_class("lr_scheduler.LinearLR", ".optim", "torch")

    print("done")
