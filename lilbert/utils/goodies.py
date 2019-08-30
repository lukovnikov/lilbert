import os
import torch
import pickle
from pathlib import Path

# Some Custom Erros
class UnknownMethod(Exception): pass
class UnknownMode(Exception): pass
class UnknownLossType(Exception): pass

create_dir = lambda dir_location: dir_location.mkdir(parents=True, exist_ok=True)


# Saving code
def save_model(model, model_name:str, output_dir:str, accuracy:list = None, config:dict = None):
    '''

    :param model:
    :param model_name:
    :param output_dir:
    :param accuracy:
    :return:
    '''

    output_dir = Path(output_dir)

    # check if the dir exists or else create it.
    create_dir(output_dir)

    torch.save(model, output_dir / model_name)


    if accuracy:
        accuracy_name = 'accuracy.pkl'
        pickle.dump(accuracy, open(output_dir / accuracy_name, 'wb+'))

    if config:
        config_name = 'config.pkl'
        pickle.dump(config, open(output_dir / config_name, 'wb+'))


class FancyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


# def save_model(model, model_name:str, output_dir:str, accuracy:list = None, config:dict = None):
#     '''
#
#     :param model:
#     :param model_name:
#     :param output_dir:
#     :param accuracy:
#     :return:
#     '''
#     counter = -1
#
#     output_dir = Path(output_dir)
#
#     # check if the dir exists or else create it.
#     create_dir(output_dir)
#
#     if not os.path.isfile(model, output_dir / model_name):
#         torch.save(model, output_dir / model_name)
#     else:
#         print(f"model name {model_name} already exists. Choosing a different one")
#         counter = 0
#         flag = True
#         while flag:
#             new_model_name = model_name + '_' + str(counter)
#             if not os.path.isfile(model, output_dir / new_model_name):
#                 torch.save(model, output_dir / new_model_name)
#                 flag = False
#             else:
#                 counter = counter + 1
#
#     if accuracy:
#         accuracy_name = 'accuracy.pkl'
#         if counter != -1:
#             accuracy_name = 'accuracy_' + str(counter) + '.pkl'
#
#         pickle.dump(accuracy, open(output_dir / accuracy_name, 'wb+'))
#
#     if config:
#         config_name = 'config.pkl'
#         if counter != -1:
#             config_name = 'config_' + str(counter) + 'pkl'
#         pickle.dump(config, open(output_dir / config_name, 'wb+'))