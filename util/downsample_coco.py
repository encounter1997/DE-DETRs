import json
import copy
import random
import argparse
import pdb


def parse_args():
    parser = argparse.ArgumentParser('Random down-sample json file', add_help=False)
    parser.add_argument('--input_file', default='instances_train2017.json', type=str)
    parser.add_argument('--sample_num', default=None, type=int)
    parser.add_argument('--sample_rate', default=0.01, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert None in [args.sample_num, args.sample_rate], \
        "args.sample_num and args.sample_rate are mutually exclusive args"

    # load data
    print('loading data')
    dataset = json.load(open(args.input_file, 'r'))
    total_sample = len(dataset['images'])

    # get sample ids
    print('generating smaple ids')
    imgid_list = range(total_sample)
    sample_num = int(total_sample * args.sample_rate) \
        if args.sample_num is None else args.sample_num
    sample_ids = random.sample(imgid_list, sample_num)

    # create sampled dataset
    print('sampling {} / {} images'.format(sample_num, total_sample))
    dataset_new = copy.deepcopy(dataset)
    dataset_new['images'] = [dataset['images'][idx] for idx in sample_ids]
    assert len(dataset_new['images']) == sample_num

    # save file
    output_file = args.input_file.replace('.json', '_sample{}.json').format(sample_num)
    print('saving to {}'.format(output_file))
    with open(output_file, 'w') as f:
        json.dump(dataset_new, f)
