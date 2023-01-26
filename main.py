import os
import argparse
import parameters
from batch_feature_extraction import extract_feature
from trainer import train
from test_fr import test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fr", default=False, action="store_true")
    parser.add_argument("--full_rank", default=False, action="store_true")
    parser.add_argument("--type_", default=2, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--pct", default=0.0, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test_model_name", default="", type=str)
    args = parser.parse_args()
    params = parameters.get_params()
    params["mode"] = args.type_
    params["pct"] = args.pct
    params["nb_epochs"] = args.epochs
    params["seed"] = args.seed
    params["batch_size"] = args.batch_size
    params["type_"] = args.type_
    params["full_rank_model_name"] = args.test_model_name
    if args.full_rank:
        params['feat_label_dir'] = os.path.join(params['dataset_dir'], 'seld_full_rank_feat_label')
        params["pct"] = 0
        params['type_'] = 1
        extract_feature(params=params)
        train(params=params)
        exit()
    # extract_feature(params=params)
    if args.test_fr:
        test(params)
    else:
        train(params=params)


if __name__ == '__main__':
    main()