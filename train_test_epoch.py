import os
import random
import numpy as np
import torch
import parameters
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2 * nb_classes], accdoa_in[:, :, 2 * nb_classes:]
    sed = np.sqrt(x ** 2 + y ** 2 + z ** 2) > 0.5

    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1 * nb_classes], accdoa_in[:, :, 1 * nb_classes:2 * nb_classes], accdoa_in[:, :,
                                                                                                   2 * nb_classes:3 * nb_classes]
    sed0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2) > 0.5
    doa0 = accdoa_in[:, :, :3 * nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3 * nb_classes:4 * nb_classes], accdoa_in[:, :,
                                                                 4 * nb_classes:5 * nb_classes], accdoa_in[:, :,
                                                                                                 5 * nb_classes:6 * nb_classes]
    sed1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) > 0.5
    doa1 = accdoa_in[:, :, 3 * nb_classes: 6 * nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6 * nb_classes:7 * nb_classes], accdoa_in[:, :,
                                                                 7 * nb_classes:8 * nb_classes], accdoa_in[:, :,
                                                                                                 8 * nb_classes:]
    sed2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2) > 0.5
    doa2 = accdoa_in[:, :, 6 * nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt + 1 * nb_classes],
                                                  doa_pred0[class_cnt + 2 * nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt + 1 * nb_classes],
                                                  doa_pred1[class_cnt + 2 * nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch_1(data_generator, model, criterion, dcase_output_folder, params, device):
    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, target in data_generator.generate():
            # load one batch of data
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            # process the batch of data based on chosen mode
            output = model(data)
            loss = criterion(output, target)
            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt],
                                                                doa_pred1[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt],
                                                                doa_pred2[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt],
                                                                doa_pred0[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append(
                                [class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + params['unique_classes']],
                                 doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + params['unique_classes']],
                                                           doa_pred[frame_cnt][
                                                               class_cnt + 2 * params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss /= nb_test_batches

    return test_loss


def train_epoch_1(data_generator, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for data, target in data_generator.generate():
        # load one batch of data
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches

    return train_loss


def test_epoch_2(data_generator, model, criterion, dcase_output_folder, params, device):
    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, target in data_generator.generate():
            # load one batch of data
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            # process the batch of data based on chosen mode
            output, _, _, _ = model(data)
            loss = criterion(output, target)
            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt],
                                                                doa_pred1[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt],
                                                                doa_pred2[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt],
                                                                doa_pred0[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append(
                                [class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + params['unique_classes']],
                                 doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + params['unique_classes']],
                                                           doa_pred[frame_cnt][
                                                               class_cnt + 2 * params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss /= nb_test_batches

    return test_loss


def train_epoch_2(data_generator, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for data, target in data_generator.generate():
        # load one batch of data
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output, _, _, _ = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches

    return train_loss


def test_epoch_3(data_generator, model, interpolator, criterion, criterion_dci, dcase_output_folder, params, device):
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss, test_dci_loss = 0, 0, 0
    model.eval()
    interpolator.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, actual_data, target in data_generator.generate():
            # load one batch of data
            data = torch.tensor(data).to(device).float()
            actual_data = torch.tensor(actual_data).to(device).float()
            target = torch.tensor(target).to(device).float()
            # process the batch of data based on chosen mode
            interpolated_data = interpolator(data)
            denoiser_loss = criterion_dci(interpolated_data, actual_data)
            denoiser_loss += denoiser_loss.item()
            out, attention, channel_map, spatial_map = model.early_attention(data)
            mapped_data = out + torch.mul(1 - spatial_map, interpolated_data)
            output = model.downstream_task(mapped_data)
            loss = criterion(output, target)
            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt],
                                                                doa_pred1[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt],
                                                                doa_pred2[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt],
                                                                doa_pred0[frame_cnt], class_cnt, params['thresh_unify'],
                                                                params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + params['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + params['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * params['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + params['unique_classes']],
                                                               doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append(
                                [class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + params['unique_classes']],
                                 doa_pred_fc[class_cnt + 2 * params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + params['unique_classes']],
                                                           doa_pred[frame_cnt][
                                                               class_cnt + 2 * params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            test_dci_loss += denoiser_loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss /= nb_test_batches

    return test_loss, test_dci_loss


def train_epoch_3(data_generator, optimizer, optimizer_dci, model, interpolator, criterion, criterion_dci, params, device):
    nb_train_batches, train_loss, dci_loss_ = 0, 0, 0
    model.train()
    interpolator.train()
    for data, actual_data, target in data_generator.generate():
        # load one batch of data
        data = torch.tensor(data).to(device).float()
        actual_data = torch.tensor(actual_data).to(device).float()
        target = torch.tensor(target).to(device).float()
        optimizer_dci.zero_grad()
        interpolated_data = interpolator(data)
        out, attention, channel_map, spatial_map = model.early_attention(data)
        # mapped_data = out.clone() + (1 - spatial_map)*denoised_data.clone()
        output = model.downstream_task(out + (1-spatial_map)*interpolated_data)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        denoiser_loss = criterion_dci(interpolated_data, actual_data)
        denoiser_loss.backward(retain_graph=True)
        dci_loss_ += denoiser_loss.item()
        optimizer.zero_grad()
        optimizer_dci.step()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break
    train_loss /= nb_train_batches
    dci_loss_ /= nb_train_batches
    return train_loss, dci_loss_