from functools import *

import ast
import csv
import logging
import math
import os
import sys
TEST_SIZE = 6

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

def prepare_train_test_data(data):
    data_size = len(data['test_data'])
    train_size = int(data_size - (data_size / TEST_SIZE))
    train_data_row = data['test_data'][0:train_size]
    test_data_row = data['test_data'][train_size:data_size]

    test_data = data.copy()
    train_data = data.copy()

    test_data['test_data'] = test_data_row
    train_data['test_data'] = train_data_row

    return train_data, test_data


def normalize_test_data(test_data):
    test_list = []
    for rows in test_data['test_data']:
        test_dict = {}
        for feature, row in zip(test_data['feature'], rows):
            test_dict[feature] = row
        test_list.append(test_dict)
    return test_list


def normalize_data(rules):
    rule_list = []
    for rule in rules:
        words = rule.split(' ')
        d = {}
        f_words = []
        for i in range(len(words)):
            if i % 2 != 0:
                f_words.append(words[i])
        d1 = {}
        cnt = len(f_words) - 1
        for j in range(len(f_words)):
            d2 = {}
            if cnt - j == 0: break
            d2[f_words[cnt - j - 1]] = f_words[cnt - j]
            if not d1:
                d1.update(d2)
            else:
                if list(d1.keys())[0] in list(d2.values()):
                    d2_k = list(d2.keys())[0]
                    d2[d2_k] = d1

            d1 = d2.copy()
        rule_list.append(d1)
    return rule_list


def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def test_tree(data, test_data, target_attribute):
    n_data = normalize_data(data)
    tree = reduce(merge, n_data)
    logger.info('TREE>>%s' % tree)
    #logger.info('test data>>%s' % test_data)
    logger.info('*************************************************************************')
    accuracy = len(test_data)
    for test in test_data:
        query = test.copy()
        target = query.pop(target_attribute)
        logger.info('target>>%s' % target)
        predicted = predict(query, tree, 1.0)
        logger.info('predicted>>%s' % predicted)
        if predicted != target:
            accuracy = accuracy-1
    accuracy = (float(accuracy)/float(len(test_data)))*100
    return accuracy


def prepare_data(filename):
    fpath = os.path.join(os.getcwd(), filename)
    fs = csv.reader(open(fpath))

    all_row = []
    for r in fs:
        all_row.append(r)

    column = all_row[0]
    id_to_name, name_to_id = get_feature_name_to_id_maps(column)

    data = {
        'feature': column,
        'test_data': all_row[1:],
        'name_to_id': name_to_id,
        'id_to_name': id_to_name
    }
    return data


def get_feature_name_to_id_maps(features):
    name_to_id = {}
    id_to_name = {}
    for i in range(0, len(features)):
        name_to_id[features[i]] = i
        id_to_name[i] = features[i]
    return id_to_name, name_to_id


def project_feature(data, columns_to_project):
    data_h = list(data['feature'])
    data_r = list(data['test_data'])

    all_cols = list(range(0, len(data_h)))

    columns_to_project_ix = [data['name_to_id'][name] for name in columns_to_project]
    columns_to_remove = [cidx for cidx in all_cols if cidx not in columns_to_project_ix]

    for delc in sorted(columns_to_remove, reverse=True):
        del data_h[delc]
        for r in data_r:
            del r[delc]

    id_to_name, name_to_id = get_feature_name_to_id_maps(data_h)

    return {'feature': data_h, 'test_data': data_r,
            'name_to_id': name_to_id,
            'id_to_name': id_to_name}


def find_feature_values(data):
    """
    find all possible combinations of feature set from the input data
    :param data:
    :return: feature_value_map
    """
    id_to_name = data['id_to_name']
    idxs = id_to_name.keys()

    feature_value_map = {}
    for idx in iter(idxs):
        feature_value_map[id_to_name[idx]] = set()

    for data_row in data['test_data']:
        for idx in id_to_name.keys():
            att_name = id_to_name[idx]
            val = data_row[idx]
            if val not in feature_value_map.keys():
                feature_value_map[att_name].add(val)
    return feature_value_map


def get_target_decision(data, target_attribute):
    """
    Given the data set and target attributes, get all decision points.
    Ex: Yes:count, No:count
    :param data:
    :param target_attribute:
    :return:
    """
    rows = data['test_data']
    col_id = data['name_to_id'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_id]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1
    return labels


def calculate_entropy(n, labels):
    ent = 0
    for label in labels.keys():
        p_x = labels[label] / n
        ent += - p_x * math.log(p_x, 2)
    return ent


def _partition(data, group_att):
    partitions = {}
    data_rows = data['test_data']
    partition_att_idx = data['name_to_id'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_id': data['name_to_id'],
                'id_to_name': data['id_to_name'],
                'test_data': list()
            }
        partitions[row_val]['test_data'].append(row)
    return partitions


def cal_avg_entropy_w_partitions(data, splitting_att, target_attribute):
    data_rows = data['test_data']
    n = len(data_rows)
    partitions = _partition(data, splitting_att)

    avg_ent = 0

    for partition_key in partitions.keys():
        partitioned_data = partitions[partition_key]
        partition_n = len(partitioned_data['test_data'])
        partition_labels = get_target_decision(partitioned_data, target_attribute)
        partition_entropy = calculate_entropy(partition_n, partition_labels)
        avg_ent += partition_n / n * partition_entropy

    return avg_ent, partitions


def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl


def id3(data, feature_elements, features, target_attribute):
    """
    A recursive algorithm to go over the data sets and find the max Information gain of features wrt target attribute and
    build the decision tree
    :param data: Test Data
    :param feature_elements: A key-value map of Feature Sets (basically all combination a feature can take)
    :param features: Feature set our algorithm is working on, this keeps getting reduced as and
    when we process them and assign to node to build the tree
    :param target_attribute: This is the feature we are trying to decision against.
    :return: A tree node
    """
    labels = get_target_decision(data, target_attribute)

    tree_node = {}
    # We are at the leaf node, return node
    if len(labels.keys()) == 1:
        tree_node['label'] = next(iter(labels.keys()))
        return tree_node

    # We processed all attributes, return node
    if len(features) == 0:
        tree_node['label'] = most_common_label(labels)
        return tree_node

    n = len(data['test_data'])
    ent = calculate_entropy(n, labels)

    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None

    for feature in features:
        avg_ent, partitions = cal_avg_entropy_w_partitions(data, feature, target_attribute)
        info_gain = ent - avg_ent
        if max_info_gain is None or info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_att = feature
            max_info_gain_partitions = partitions

    if max_info_gain is None:
        tree_node['label'] = most_common_label(labels)
        return tree_node

    tree_node['attribute'] = max_info_gain_att
    tree_node['nodes'] = {}

    remaining_atts_for_subtrees = set(features)
    remaining_atts_for_subtrees.discard(max_info_gain_att)

    feature_att_values = feature_elements[max_info_gain_att]

    for att_value in feature_att_values:
        if att_value not in max_info_gain_partitions.keys():
            tree_node['nodes'][att_value] = {'label': most_common_label(labels)}
            continue
        partition = max_info_gain_partitions[att_value]
        value = id3(partition, feature_elements, remaining_atts_for_subtrees, target_attribute)
        tree_node['nodes'][att_value] = value

    return tree_node


def load_input_config(config_file):
    with open(config_file, 'r') as data_file:
        data = data_file.read().replace('\n', '')
    return ast.literal_eval(data)


def print_tree(root, tree):
    stack = []
    rules = set()

    def traverse(node, stack, rules):
        if 'label' in node:
            stack.append(' THEN ' + node['label'])
            rules.add(''.join(stack))
            stack.pop()
        elif 'attribute' in node:
            ifnd = 'IF ' if not stack else ' AND '
            stack.append(ifnd + node['attribute'] + ' EQUALS ')
            for subnode_key in node['nodes']:
                stack.append(subnode_key)
                traverse(node['nodes'][subnode_key], stack, rules)
                stack.pop()
            stack.pop()

    traverse(root, stack, rules)
    print(os.linesep.join(rules))
    return rules


def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def decision_tree():
    argv = sys.argv
    logger.info('**************************************')
    logger.info("COMMAND LINE ARGS {}: ".format(argv))

    config = load_input_config(argv[1])

    # prepare data from the input test file
    data = prepare_data(config['data_file'])
    train_data, test_data = prepare_train_test_data(data)
    logger.info('*************************************************************************')
    logger.info('TRAINING DATA: {} SIZE: {}'.format(train_data, len(train_data['test_data'])))
    logger.info('*************************************************************************')
    logger.info('TEST DATA: {} SIZE: {}'.format(test_data, len(test_data['test_data'])))

    test_data = normalize_test_data(test_data)

    train_data = project_feature(train_data, config['data_project_columns'])

    target_attribute = config['target_attribute']
    features = set(train_data['feature'])
    features.remove(target_attribute)

    feature_elements = find_feature_values(train_data)

    root = id3(train_data, feature_elements, features, target_attribute)

    tree = []
    logger.info('*************************************************************************')
    logger.info('DECISION:>>')
    rules = print_tree(root, tree)

    logger.info('*************************************************************************')
    logger.info('TEST DATA >> %s' % test_data)
    logger.info('*************************************************************************')
    accuracy = test_tree(rules, test_data, target_attribute)
    logger.info('*************************************************************************')
    logger.info('ACCURACY:%s', accuracy)
    logger.info('*************************************************************************')


if __name__ == "__main__":
    decision_tree()
