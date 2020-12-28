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


# ========= Question 2

def pages_in_clicks(graph, initial_page, num_clicks, print_=False):
    """
    Given a graph, an initial starting point and the number of clicks, how many, and which pages will we be able to
    visit?

    :param graph: Graph  over which the analysis will be done (graph needs to be provided in default dictionary format)
    {108:[1059989, 1062426, 1161925], 95:[1185516]}
    Article 108 is linked to the articles 1059989, 1062426, 1161925
    :param initial_page: Page we will be starting out from
    :param num_clicks: Number of clicks we are willing to do
    :param print_: Bool to visualize some of the outputs or not
    :return: Pages seen  with the given number of clicks
    """

    # This will be a list of all the pages that we are able to visit during our clicks
    pages_seen = []
    # This will be a list of articles that we are able to reach at the ith click. We will use this list to check the
    # articles that we can reach in the i+1th click
    queue = [initial_page]

    # Placeholder to keep track of the clicks we are doing
    clicks = 0
    # Interrupt the loop once we reach the required number of clicks
    while clicks < num_clicks:
        # List of elements that will be used for the next loop
        new_queue = []
        # List of elements that don't have any out-node
        last_nodes = []
        # Loop over all the pages of the current click
        for node in queue:
            if print_:
                print(node)
            # If a given node has target node, include the out-nodes into the new_queue list
            if graph[node] != 0:
                new_queue += graph[node]
            # If a given node has no target node, include it in the last_nodes list (this list will not be used to)
            # for further inspection but we will have to consider it as an article that has been seen
            else:
                last_nodes.append(node)

        if print_:
            print(new_queue)

        # Update queue as the new pages to explore
        queue = new_queue
        # Update pages_seen with the pages that have been seen in this click
        pages_seen += new_queue + last_nodes
        # Update the number of clicks done
        clicks += 1

    # Return the unique pages
    return set(pages_seen)
