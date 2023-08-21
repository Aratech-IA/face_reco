# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 11:11:54 2019

@author: julien
"""
import logging
from logging.handlers import RotatingFileHandler
import os
from django.conf import settings

# ------------------------------------------------------------------------------
# a simple config to create a file log - change the level to warning in
# production
# ------------------------------------------------------------------------------


class Logger(object):
    def __init__(self, name, level=logging.ERROR, file=False):
        self.logger = logging.getLogger(name)
        if not len(self.logger.handlers):
            self.logger.setLevel(level)
            formatter = logging.Formatter('%(name)s :: %(asctime)s :: %(levelname)s :: %(message)s')
            if file:
                file_handler = RotatingFileHandler(os.path.join(settings.BASE_DIR, 'projet/settings_clients/log',
                                                                name+'.log'), 'a', 10000000, 1)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.propagate = False

    def run(self):
        return self.logger


"""
def make_logger(name, level='ERROR'):
    log_level = getattr(logging, level)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(os.path.join(settings.BASE_DIR, 'log', name + '.log'), 'a', 10000000, 1)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
"""
