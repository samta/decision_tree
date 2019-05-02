import ast
import csv
import sys
import math
import os
from functools import *
TEST_SIZE = 6


def load_csv_to_header_data(filename):
    fpath = os.path.join(os.getcwd(), filename)
    fs = csv.reader(open(fpath))

    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)

    data = {
        'header': headers,
        'rows': all_row[1:],
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }
    return data


def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx


def project_columns(data, columns_to_project):
    data_h = list(data['header'])
    data_r = list(data['rows'])

    all_cols = list(range(0, len(data_h)))

    columns_to_project_ix = [data['name_to_idx'][name] for name in columns_to_project]
    columns_to_remove = [cidx for cidx in all_cols if cidx not in columns_to_project_ix]

    for delc in sorted(columns_to_remove, reverse=True):
        del data_h[delc]
        for r in data_r:
            del r[delc]

    idx_to_name, name_to_idx = get_header_name_to_idx_maps(data_h)

    return {'header': data_h, 'rows': data_r,
            'name_to_idx': name_to_idx,
            'idx_to_name': idx_to_name}


def get_uniq_values(data):
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()

    val_map = {}
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]
            if val not in val_map.keys():
                val_map[att_name].add(val)
    return val_map


def get_class_labels(data, target_attribute):
    rows = data['rows']
    col_idx = data['name_to_idx'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_idx]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1
    return labels


def entropy(n, labels):
    ent = 0
    for label in labels.keys():
        p_x = labels[label] / n
        ent += - p_x * math.log(p_x, 2)
    return ent


def partition_data(data, group_att):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }
        partitions[row_val]['rows'].append(row)
    return partitions


def avg_entropy_w_partitions(data, splitting_att, target_attribute):
    # find uniq values of splitting att
    data_rows = data['rows']
    n = len(data_rows)
    partitions = partition_data(data, splitting_att)

    avg_ent = 0

    for partition_key in partitions.keys():
        partitioned_data = partitions[partition_key]
        partition_n = len(partitioned_data['rows'])
        partition_labels = get_class_labels(partitioned_data, target_attribute)
        partition_entropy = entropy(partition_n, partition_labels)
        avg_ent += partition_n / n * partition_entropy

    return avg_ent, partitions


def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl


def id3(data, uniqs, remaining_atts, target_attribute):
    labels = get_class_labels(data, target_attribute)

    node = {}
    if len(labels.keys()) == 1:
        node['label'] = next(iter(labels.keys()))
        return node

    if len(remaining_atts) == 0:
        node['label'] = most_common_label(labels)
        return node

    n = len(data['rows'])
    ent = entropy(n, labels)

    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None

    for remaining_att in remaining_atts:
        avg_ent, partitions = avg_entropy_w_partitions(data, remaining_att, target_attribute)
        info_gain = ent - avg_ent
        if max_info_gain is None or info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_att = remaining_att
            max_info_gain_partitions = partitions

    if max_info_gain is None:
        node['label'] = most_common_label(labels)
        return node

    node['attribute'] = max_info_gain_att
    node['nodes'] = {}

    remaining_atts_for_subtrees = set(remaining_atts)
    remaining_atts_for_subtrees.discard(max_info_gain_att)

    uniq_att_values = uniqs[max_info_gain_att]

    for att_value in uniq_att_values:
        if att_value not in max_info_gain_partitions.keys():
            node['nodes'][att_value] = {'label': most_common_label(labels)}
            continue
        partition = max_info_gain_partitions[att_value]
        value = id3(partition, uniqs, remaining_atts_for_subtrees, target_attribute)
        node['nodes'][att_value] = value

    return node


def load_config(config_file):
    with open(config_file, 'r') as myfile:
        data = myfile.read().replace('\n', '')
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
    # 1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            # 2.
            try:
                result = tree[key][query[key]]
            except:
                return default
            # 3.
            result = tree[key][query[key]]
            # 4.
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def prepare_train_test_data(data):
    data_size = len(data['rows'])
    train_size = int(data_size - (data_size / TEST_SIZE))
    train_data_row = data['rows'][0:train_size]
    test_data_row = data['rows'][train_size:data_size]

    test_data = data.copy()
    train_data = data.copy()

    test_data['rows'] = test_data_row
    train_data['rows'] = train_data_row

    return train_data, test_data


def normalize_test_data(test_data):
    test_list = []
    for rows in test_data['rows']:
        test_dict = {}
        for header, row in zip(test_data['header'], rows):
            test_dict[header] = row
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
    print('tree>>', tree)
    print('test data>>', test_data)
    accuracy = len(test_data)
    for test in test_data:
        query = test.copy()
        target = query.pop(target_attribute)
        print ('target>>', target)
        predicted = predict(query, tree, 1.0)
        print ('predicted>>', predicted)
        if predicted != target:
            accuracy = accuracy-1
    accuracy = (float(accuracy)/float(len(test_data)))*100
    return accuracy


def main():
    argv = sys.argv
    print ('**************************************')
    print("Command line args are {}: ".format(argv))

    config = load_config(argv[1])

    data = load_csv_to_header_data(config['data_file'])
    train_data, test_data = prepare_train_test_data(data)
    print ('**************************************')
    print ('training data: %s training data size: %s' % (train_data, len(train_data['rows'])))
    print ('**************************************')
    print ('test data: %s test data size: %s' % (test_data, len(test_data['rows'])))

    test_data = normalize_test_data(test_data)

    train_data = project_columns(train_data, config['data_project_columns'])

    target_attribute = config['target_attribute']
    remaining_attributes = set(train_data['header'])
    remaining_attributes.remove(target_attribute)

    uniqs = get_uniq_values(train_data)

    root = id3(train_data, uniqs, remaining_attributes, target_attribute)

    tree = []
    rules = print_tree(root, tree)
    print ('**************************************')
    accuracy = test_tree(rules, test_data, target_attribute)
    print ('**************************************')
    print ('Accuracy:', accuracy)
    print ('**************************************')


if __name__ == "__main__":
    main()
