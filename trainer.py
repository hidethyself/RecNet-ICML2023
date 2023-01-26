import torch
import time
import torch.nn as nn
import torch.optim as optim
import cls_data_generator
import cls_data_generator_full_rank
import cls_feature_class
from model import ModelConfig
from train_test_epoch import *


def train(params):
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
    if params["type_"] == 1:
        data_gen_train = cls_data_generator.DataGenerator(params=params, split=train_splits[0])
        data_in, data_out = data_gen_train.get_data_sizes()
        data_gen_val = cls_data_generator.DataGenerator(
            params=params,
            split=val_splits[0],
            shuffle=False,
            per_file=True
        )
    elif params["type_"] == 2:
        data_gen_train = cls_data_generator_full_rank.DataGenerator(params=params, split=train_splits[0])
        data_in, data_out = data_gen_train.get_data_sizes()
        data_gen_val = cls_data_generator_full_rank.DataGenerator(
            params=params,
            split=val_splits[0],
            shuffle=False,
            per_file=True
        )
    model_config = ModelConfig(
        type_=params["type_"],
        input_shape=data_in,
        out_shape=data_out,
        parameters=params
    )
    dcase_output_val_folder = os.path.join(params['dcase_output_dir'], 'baseline_{}_val'.format(params["seed"]))
    cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
    criterion = nn.MSELoss()
    if params["type_"] == 1:
        model_name = f'./models/baseline_{params["seed"]}_{int(params["pct"]*100)}.h5'
        text_file_name = f'./result_txt/baseline_{params["seed"]}_{int(params["pct"]*100)}.txt'
        model = model_config.load_model()
        model = model.to(device=device)
    elif params["type_"] == 2:
        model_name = f'./models/recnet_{params["seed"]}_{int(params["pct"]*100)}.h5'
        interpolator_name = f'./models/recnet_interpolator_{params["seed"]}_{int(params["pct"]*100)}.h5'
        text_file_name = f'./result_txt/recnet_{params["seed"]}_{int(params["pct"]*100)}.txt'
        model, deep_interpolator = model_config.load_model()
        model = model.to(device)
        deep_interpolator = deep_interpolator.to(device=device)
        optimizer_dci = optim.Adam(deep_interpolator.parameters(), lr=params['dci_lr'])
        criterion_dci = nn.MSELoss()
    else:
        print("select correct type_!")
        exit()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    best_val_epoch = -1
    best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999
    for epoch_cnt, epoch in enumerate(range(params["nb_epochs"])):
        if params["type_"] == 1:
            start_time = time.time()
            train_loss = train_epoch_1(
                data_generator=data_gen_train, optimizer=optimizer, model=model,
                criterion=criterion, params=params, device=device
            )
            train_time = time.time() -start_time
            start_time = time.time()
            val_loss = test_epoch_1(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
        if params["type_"] == 2:
            start_time = time.time()
            train_loss, denoiser_loss_ = train_epoch_3(
                data_gen_train, optimizer, optimizer_dci,
                model, deep_interpolator, criterion, criterion_dci,
                params, device
            )
            train_time = time.time() - start_time
            start_time = time.time()
            val_loss, val_denoise_loss_ = test_epoch_3(
                data_gen_val, model, deep_interpolator,
                criterion, criterion_dci, dcase_output_val_folder, params, device)
        val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(
            dcase_output_val_folder)
        val_time = time.time() - start_time
        if val_seld_scr <= best_seld_scr:
            best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr
            torch.save(model.state_dict(), model_name)
            if params['type_'] == 2:
                torch.save(deep_interpolator.state_dict(), interpolator_name)
        print(
            'epoch: {}, time: {:0.2f}/{:0.2f}, '
            'train_loss: {:0.4f}, val_loss: {:0.4f}, '
            'ER/F/LE/LR/SELD: {}, '
            'best_val_epoch: {} {}'.format(
                epoch_cnt, train_time, val_time,
                train_loss, val_loss,
                '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
                best_val_epoch,
                '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR,
                                                                   best_seld_scr))
        )

    print('Load best model weights')
    if params["type_"] == 1:
        model.load_state_dict(torch.load(model_name, map_location='cpu'))
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
    elif params["type_"] == 2:
        data_gen_test = cls_data_generator_full_rank.DataGenerator(
            params=params,
            split=test_splits[0],
            shuffle=False,
            per_file=True
        )
        model.load_state_dict(torch.load(model_name, map_location='cpu'))
        deep_interpolator.load_state_dict(torch.load(interpolator_name, map_location='cpu'))
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'],
                                                'RecNet_{}_test'.format(params["seed"]))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

        test_loss, test_denoiser_loss = test_epoch_3(data_gen_test, model, deep_interpolator, criterion, criterion_dci,
                                                     dcase_output_test_folder, params, device)

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
    print_0 = f'Best val epoch: {best_val_epoch}\n\n'
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
    text_file = open(text_file_name, "w")
    text_file.write(print_0 + print_00 + '\n' + print_1 + '\n\n' + print_2)
    text_file.close()
