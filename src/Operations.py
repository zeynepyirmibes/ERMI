# -*- coding: utf-8 -*-

import pickle
import logging
import os


def get_logger(f_path):
    level = logging.INFO
    format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'
    handlers = [logging.FileHandler(os.path.join(f_path, 'model.log')), logging.StreamHandler()]
    logging.basicConfig(level=level, datefmt='%Y-%m-%d %H:%M', format=format, handlers=handlers)
    logger = logging.getLogger('ERMI')
    return logger


def dump_pickle(data, pkl_path):
    print('Dumping to pickle %s ...' % pkl_path)
    with open(pkl_path, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


def load_pickle(pkl_path):
    print('Loading %s to pickle...' % pkl_path)
    return pickle.load(open(pkl_path, 'rb'))
