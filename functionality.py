import pandas as pd
import re
import os
import json
import random


def read_pages_category(filename='data/wiki-topcats-categories.txt'):
    """
    Function to read the wiki-topcats-categories.txt file

    :param filename:
    :return:
    """
    file = open(filename, 'r')
    Lines = file.readlines()

    pages_category = {}
    for count, line in enumerate(Lines):
        category = re.search(':(.*);', line).group(1)
        pages = line.split(';')[1].rsplit()
        pages_category[category] = list(map(int, pages))

    return pages_category


def read_name_page(filename='data/wiki-topcats-page-names.txt'):
    """
    Function to read the wiki-topcats-categories.txt file

    :param filename:
    :return:
    """
    file = open(filename, 'r')
    Lines = file.readlines()

    name_page = {}
    for count, line in enumerate(Lines):
        line_split = re.split(r'(?<=\d)\D', line, maxsplit=1)
        name_page[line_split[0]] = line_split[1].rstrip("\n")
    return name_page


def write_json(file_name, content):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as outfile:
        json.dump(content, outfile, sort_keys=True, indent=4)


def read_json(file_name):
    with open(file_name) as json_file:
        data_dict = json.load(json_file)
        return data_dict


def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x


def read_json_int_key(file_name):
    with open(file_name) as json_file:
        data_dict = json.load(json_file, object_hook=jsonKeys2int)
        return data_dict


def revert_dict_list(dictionary):
    """
    Function to revert the values and keys of a dictionary
    :param dictionary: Dictionary to revert
    :return: Reverted dictionary
    """
    new_dic = {}
    for k, v in dictionary.items():
        for x in v:
            new_dic.setdefault(x, []).append(k)
    return new_dic


def revert_dict(dictionary):
    """
    Function to revert the values and keys of a dictionary
    :param dictionary: Dictionary to revert
    :return: Reverted dictionary
    """
    new_dic = {}
    for k, v in dictionary.items():
        new_dic.setdefault(v, []).append(k)
    return new_dic


def uniformly_pick_article_category(dictionary):
    """
    Given a dictionary in which we have as keys the article identifier (number) and the categories it belongs to,
    randomly choose one of these categories and generate new dictionary

    :param dictionary: Dictionary with the previous features
    :return: New dict with unique category values
    """

    new_dict = {}
    for k, v in dictionary.items():
        new_dict[k] = random.choice(v)

    return new_dict
