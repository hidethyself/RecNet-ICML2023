import torch
import time
import torch.nn as nn
import torch.optim as optim
import cls_data_generator
import cls_data_generator_full_rank
import cls_feature_class
from model import ModelConfig
from train_test_epoch import *


def test(params):
    seed = params["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)
    test_splits = [6]
    val_splits = [5]
    train_splits = [[1, 2, 3, 4]]
    loc_feat = params['dataset']
    loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'
    score_obj = ComputeSELDResults(params)
    cls_feature_class.create_folder(params['model_dir'])
    data_gen_train = cls_data_generator.DataGenerator(params=params, split=train_splits[0])
    data_in, data_out = data_gen_train.get_data_sizes()
    model_config = ModelConfig(
        type_=1,
        input_shape=data_in,
        out_shape=data_out,
        parameters=params
    )
    model = model_config.load_model().to(device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    print('Load best model weights')
    model.load_state_dict(torch.load(params['full_rank_model_name'], map_location='cpu'))
    data_gen_test = cls_data_generator.DataGenerator(
        params=params,
        split=test_splits[0],
        shuffle=False,
        per_file=True
    )
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=test_splits[0], shuffle=False, per_file=True
    )
    dcase_output_test_folder = os.path.join(params['dcase_output_dir'],
                                            'baseline_{}_test'.format(params["seed"]))
    cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
    print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))
    test_loss = test_epoch_1(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
    use_jackknife = True
    test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(
        dcase_output_test_folder, is_jackknife=use_jackknife)
    print('\nTest Loss')
    print('SELD score (early stopping metric): {:0.2f} {}'.format(
        test_seld_scr[0] if use_jackknife else test_seld_scr,
        '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
    print_00 = 'SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(
        test_ER[0] if use_jackknife else test_ER,
        '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0],
                                    test_ER[1][
                                        1]) if use_jackknife else '',
        100 * test_F[
            0] if use_jackknife else 100 * test_F,
        '[{:0.2f}, {:0.2f}]'.format(
            100 * test_F[1][0], 100 * test_F[1][
                1]) if use_jackknife else '')
    print(print_00)
    print_1 = 'DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(
        test_LE[0] if use_jackknife else test_LE,
        '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '',
        100 * test_LR[0] if use_jackknife else 100 * test_LR,
        '[{:0.2f}, {:0.2f}]'.format(100 * test_LR[1][0], 100 * test_LR[1][1]) if use_jackknife else '')
    print(print_1)
    print_2 = ''
    if params['average'] == 'macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            xx = '{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                cls_cnt,
                classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                            classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                            classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                            classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                            classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                            classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '')
            print(xx)
            print_2 += (xx + '\n')
    text_file = open("./result_txt/fr_test.txt", "w")
    text_file.write(print_00 + '\n' + print_1 + '\n\n' + print_2)
    text_file.close()
