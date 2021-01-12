import pandas as pd
import re
import os
import json
import random


def read_graph(filename='data/wikigraph_reduced.csv', output='dict'):
    """
    Read graph  and determine the output required (dict or dataframe)

    :param filename: Path where the  file is saved
    :param output: Output format
    :return: Dict or Dataframe
    """

    link_df = pd.read_csv(filename, sep='\t', index_col=0)
    link_df.columns = ['Source', 'Target']

    if output == 'df':
        return link_df
    else:
        link_dict = {}
        for index, row in link_df.iterrows():
            link_dict.setdefault(row[0], []).append(row[1])
        return link_dict


def read_pages_category(filename='data/wiki-topcats-categories.txt'):
    """
    Function to read the wiki-topcats-categories.txt file

    :param filename:
    :return:
    """
    file = open(filename, 'r')
    lines = file.readlines()

    pages_category = {}
    for line in lines:
        category = re.search(':(.*);', line).group(1)
        pages = line.split(';')[1].rsplit()
        pages_category[category] = list(map(int, pages))

    return pages_category


def read_graph_category(g, filename='data/graph_pages_category.json'):
    """
    Function to read the category:[graph vertices] file

    :param filename:
    :return: category dictionary
    """
    if not os.path.isfile(filename):
        categories = read_json('data/final_pages_category.json')
        vert = set(g.get_vertices())
        cat = list(categories.keys())
        for c in cat:
            categories[c] = [v for v in categories[c] if v in vert]
            if len(categories[c]) == 0:
                categories.pop(c)
        write_json(filename, categories)
    return read_json(filename)


def read_name_page(filename='data/wiki-topcats-page-names.txt'):
    """
    Function to read the wiki-topcats-categories.txt file

    :param filename:
    :return:
    """
    file = open(filename, 'r')
    lines = file.readlines()

    name_page = {}
    for line in lines:
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


def apply_category_constraint(category_dict):
    """
    Remove the categories that have less than 5000 pages or more than 30000

    :param category_dict : dictionary of categories
    :return : filtered dictionary of categories
    """
    categories = list(category_dict.keys())
    for c in categories:
        if not (5000 <= len(category_dict[c]) <= 30000):
            category_dict.pop(c)
    return category_dict