"""WikiQA data loader."""

import typing
import csv
from pathlib import Path

import pandas as pd

import matchzoo
from matchzoo.engine.base_task import BaseTask


def read_data(
    data_root: str = "../../data",
    stage: str = 'train',
    task: typing.Union[str, BaseTask] = 'ranking',
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load local data.

    :param data_root: the root path of data
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classification` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = Path(data_root)
    file_path = data_root.joinpath(f'{stage}.csv')
    data_pack = _read_data(file_path, task)
    if task == 'ranking' or isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif task == 'classification' or isinstance(
            task, matchzoo.tasks.Classification):
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data(url, name="wikiQA"):
    ref_path = matchzoo.utils.get_file(
        name, url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir=name
    )
    return Path(ref_path).parent.joinpath("".join([name, "Corpus"]))


def _read_data(path, task):
    table = pd.read_csv(path, sep=',', header=0, quoting=csv.QUOTE_NONE)
    if "text_right" not in table.columns:
        table["text_right"] = table["text_left"]
    df = pd.DataFrame({
        'text_left': table['text_left'],
        'text_right': table['text_right'],
        # 'id_left': table['QuestionID'],
        # 'id_right': table['SentenceID'],
        'label': table['Label']
    })
    return matchzoo.pack(df, task)

