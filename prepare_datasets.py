import argparse
import numpy as np
import os
import pandas as pd
import datetime
import json

'''
 start2end_dict={ # [start, end]
        'PEMS04':'2018-01-01 00:00 to 2018-02-28 23:55', #5min
        'PEMS07':'2017-05-01 00:00 to 2017-08-06 23:55', #5min
        'PEMS08':'2016-07-01 00:00 to 2016-08-31 23:55', #5min
        'T-Drive':'2015-02-01 00:00 to 2015-06-30 23:00', #60min
        'NYCTaxi':'2014-01-01 00:00 to 2014-12-31 23:30', #30min
        'CHIBike':'2020-07-01 00:00 to 2020-09-30 23:30', #30min
    }
'''
start_time_dict = {  # [start,end)
    'PEMS04': '2018-01-01 00:00:00',  # 5min
    'PEMS07': '2017-05-01 00:00:00',  # 5min
    'PEMS08': '2016-07-01 00:00:00',  # 5min
    'T-Drive': '2015-02-01 00:00:00',  # 60min
    'NYCTaxi': '2014-01-01 00:00:00',  # 30min
    'CHIBike': '2020-07-01 00:00:00',  # 30min
}
end_time_dict = {  # [start,end)
    'PEMS04': '2018-03-01 00:00:00',  # 5min
    'PEMS07': '2017-08-07 00:00:00',  # 5min
    'PEMS08': '2016-09-01 00:00:00',  # 5min
    'T-Drive': '2015-07-01 00:00:00',  # 60min
    'NYCTaxi': '2015-01-01 00:00:00',  # 30min
    'CHIBike': '2020-10-01 00:00:00',  # 30min
}
time_interval_dict = {
    'PEMS04': 5,  # 5min
    'PEMS07': 5,  # 5min
    'PEMS08': 5,  # 5min
    'T-Drive': 60,  # 60min
    'NYCTaxi': 30,  # 30min
    'CHIBike': 30,  # 30min
}


def load_grid(filename):
    print("Loading file " + filename)
    gridfile = pd.read_csv(filename)
    df = gridfile[['inflow', 'outflow']]
    max_row_id = gridfile['row_id'].max()
    max_col_id = gridfile['column_id'].max()
    num_nodes = (max_row_id+1)*(max_col_id+1)

    timesolts = list(gridfile['time'][:int(gridfile.shape[0] / num_nodes)])
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i + len_time].values)
    data = np.array(data, dtype=float)
    data = data.swapaxes(0, 1)

    # if not os.path.exists(f'./datasets/{filename}.npz'):
    #     print('save data into npz file:')
    #     np.savez_compressed(f'./datasets/{filename}.npz', data=data)
    print("Loaded file " + filename + '.grid' + ', shape=' + str(data.shape))
    return data


def load_dyna(filename):
    print("Loading file " + filename)
    dynafile = pd.read_csv(filename)
    df = dynafile[['traffic_flow']]
    num_nodes = dynafile['entity_id'].max()+1

    timesolts = list(dynafile['time'][:int(dynafile.shape[0] / num_nodes)])
    len_time = len(timesolts)
    data = []
    for i in range(0, df.shape[0], len_time):
        data.append(df[i:i+len_time].values)
    data = np.array(data, dtype=np.float)
    data = data.swapaxes(0, 1)
    print("Loaded file " + filename + '.dyna' + ', shape=' + str(data.shape))
    return data


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets, dataset_name='PEMS08', slice_size_per_day=288, timestamp_np=None, time_delta=None
):
    num_samples, num_nodes, in_dim = data.shape
    print(data.shape)
    if timestamp_np is not None:
        start_dt = timestamp_np[0].astype(datetime.datetime)
        time_start_offset = start_dt - \
            datetime.datetime(year=start_dt.year,
                              month=start_dt.month, day=start_dt.day)
        print(timestamp_np[0])
        assert time_start_offset % time_delta == datetime.timedelta(hours=0)
        slice_start = int(time_start_offset / time_delta)
        dayofweek_start = timestamp_np[0].astype(datetime.datetime).weekday()
    else:
        # the start slice id of the day
        slice_start_dict = {'PEMS03': 0, 'PEMS04': 0, 'PEMS07': 0, 'PEMS08': 0,
                            'HZME_INFLOW': 0, 'HZME_OUTFLOW': 0,
                            'T-Drive': 0, 'NYCTaxi': 0, 'CHIBike': 0}
        slice_start = slice_start_dict[dataset_name]
        print('slice_start:', slice_start)
        # the day of the week of the beginning day
        dayofweek_start_dict = {'PEMS03': 5, 'PEMS04': 0, 'PEMS07': 0, 'PEMS08': 4,
                                'HZME_INFLOW': 1, 'HZME_OUTFLOW': 1,
                                'T-Drive': 6, 'NYCTaxi': 2, 'CHIBike': 2}
        dayofweek_start = dayofweek_start_dict[dataset_name]
    print('slice_start:', slice_start)
    print('dayofweek_start:', dayofweek_start)

    tod = [(i + slice_start) % slice_size_per_day for i in range(num_samples)]
    t_of_d = np.array(tod)
    t_of_d = np.expand_dims(t_of_d, 1)
    print('t_of_d.shape:', t_of_d.shape)

    dow = [(i // slice_size_per_day + dayofweek_start) %
           7 for i in range(num_samples)]
    d_of_w = np.array(dow)
    d_of_w = np.expand_dims(d_of_w, 1)
    print('d_of_w.shape:', d_of_w.shape)

    x, y = [], []
    x_tod, y_tod = [], []
    x_dow, y_dow = [], []
    if timestamp_np is not None:
        x_timestamp, y_timestamp = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):  # t is the index of the latest observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        x_tod.append(t_of_d[t + x_offsets, ...])
        y_tod.append(t_of_d[t + y_offsets, ...])
        x_dow.append(d_of_w[t + x_offsets, ...])
        y_dow.append(d_of_w[t + y_offsets, ...])
        if timestamp_np is not None:
            x_timestamp.append(timestamp_np[t + x_offsets])
            y_timestamp.append(timestamp_np[t + y_offsets])

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    x_tod = np.stack(x_tod, axis=0)
    y_tod = np.stack(y_tod, axis=0)
    x_dow = np.stack(x_dow, axis=0)
    y_dow = np.stack(y_dow, axis=0)
    if timestamp_np is not None:
        x_timestamp = np.stack(x_timestamp, axis=0)
        y_timestamp = np.stack(y_timestamp, axis=0)
    else:
        x_timestamp = None
        y_timestamp = None
    return x, y, x_tod, y_tod, x_dow, y_dow, x_timestamp, y_timestamp


def data_for_short_term_forecasting(dataset_name, data_file, input_length, predict_length, output_dir,
                                    save=False, save_timestamp_np=False, save_description=False,
                                    train_ratio=0.6, test_ratio=0.2,
                                    slice_size_per_day=288, split_type='round'):
    y_start = 1
    if data_file.endswith('npz'):
        ori = np.load(data_file)
        data = ori['data'][..., 0:1]
    elif data_file.endswith('grid'):
        data = load_grid(data_file)
    elif data_file.endswith('dyna'):
        data = load_dyna(data_file)
    print(data.shape)

    num_time_steps = data.shape[0]
    start_datetime = datetime.datetime.strptime(
        start_time_dict[dataset_name], '%Y-%m-%d %H:%M:%S')
    time_delta = datetime.timedelta(minutes=time_interval_dict[dataset_name])
    timestamp_list = []
    dt = start_datetime
    for i in range(num_time_steps):
        timestamp_list.append(np.datetime64(dt))
        dt += time_delta
    timestamp_np = np.array(timestamp_list)
    print(timestamp_np.shape)
    if save_timestamp_np:
        np.save(os.path.join(
            output_dir, f"{dataset_name}_timestamp.npy"), timestamp_np)
        print(os.path.join(output_dir,
              f"{dataset_name}_timestamp.npy")+' saved.')
    # print(timestamp_np.dtype)
    # print(timestamp_np[0:10])

    x_offsets = np.sort(np.concatenate(
        (np.arange(-(input_length - 1), 1, 1),)))
    print('x_offsets:', x_offsets)
    # Predict the next one hour
    y_offsets = np.sort(np.arange(y_start, (predict_length + 1), 1))
    print('y_offsets:', y_offsets)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)

    x, y, x_tod, y_tod, x_dow, y_dow, x_timestamp, y_timestamp = generate_graph_seq2seq_io_data(
        data,
        timestamp_np=timestamp_np,
        time_delta=time_delta,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        dataset_name=dataset_name,
        slice_size_per_day=slice_size_per_day
    )

    print('data shape:', data.shape)
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    print('x_tod.shape: ', x_tod.shape, ',y_tod.shape: ', y_tod.shape)
    print('x_dow.shape: ', x_dow.shape, ',y_dow.shape: ', y_dow.shape)
    print('x_timestamp.shape: ', x_timestamp.shape,
          ',y_timestamp.shape: ', y_timestamp.shape)

    num_samples = x.shape[0]
    assert split_type in ['int', 'round']
    print('split_type', split_type)
    if split_type == 'int':
        split_line1 = int(num_samples * (train_ratio))
        split_line2 = int(num_samples * (1-test_ratio))
    else:
        split_line1 = round(num_samples * (train_ratio))
        split_line2 = round(num_samples * (1-test_ratio))

    def split_data(data, line1, line2):
        data_train = data[:line1]
        data_val = data[line1:line2]
        data_test = data[line2:]
        return data_train, data_val, data_test
    x_train, x_val, x_test = split_data(x, split_line1, split_line2)
    y_train, y_val, y_test = split_data(y, split_line1, split_line2)
    x_tod_train, x_tod_val, x_tod_test = split_data(
        x_tod, split_line1, split_line2)
    x_dow_train, x_dow_val, x_dow_test = split_data(
        x_dow, split_line1, split_line2)
    y_tod_train, y_tod_val, y_tod_test = split_data(
        y_tod, split_line1, split_line2)
    y_dow_train, y_dow_val, y_dow_test = split_data(
        y_dow, split_line1, split_line2)
    x_timestamp_train, x_timestamp_val, x_timestamp_test = split_data(
        x_timestamp, split_line1, split_line2)
    y_timestamp_train, y_timestamp_val, y_timestamp_test = split_data(
        y_timestamp, split_line1, split_line2)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _x_tod = locals()['x_tod_'+cat]
        _x_dow = locals()['x_dow_'+cat]
        _y_tod = locals()['y_tod_'+cat]
        _y_dow = locals()['y_dow_'+cat]
        _x_timestamp = locals()['x_timestamp_'+cat]
        _y_timestamp = locals()['y_timestamp_'+cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        print(cat, "x_tod:", _x_tod.shape, "x_dow:", _x_dow.shape)
        print(cat, "y_tod:", _y_tod.shape, "y_dow:", _y_dow.shape)
        print(cat, 'x_timestamp:', _x_timestamp.shape)
        print(cat, 'y_timestamp:', _y_timestamp.shape)

    if save:
        output_path = os.path.join(output_dir, f"{dataset_name}_{input_length}to{predict_length}.npz")
        print(f"save file to {output_path}")
        np.savez_compressed(
            output_path,
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            x_tod_train=x_tod_train, x_tod_val=x_tod_val, x_tod_test=x_tod_test,
            x_dow_train=x_dow_train, x_dow_val=x_dow_val, x_dow_test=x_dow_test,
            y_tod_train=y_tod_train, y_dow_train=y_dow_train, y_tod_val=y_tod_val,
            y_dow_val=y_dow_val, y_tod_test=y_tod_test, y_dow_test=y_dow_test
        )
        np.savez_compressed(
            os.path.join(
                output_dir, f"{dataset_name}_{input_length}to{predict_length}_timestamp.npz"),
            x_timestamp_train=x_timestamp_train,
            x_timestamp_val=x_timestamp_val,
            x_timestamp_test=x_timestamp_test,
            y_timestamp_train=y_timestamp_train,
            y_timestamp_val=y_timestamp_val,
            y_timestamp_test=y_timestamp_test
        )
        print('npz file has been generated!')

    if save or save_description:
        with open(os.path.join(output_dir, f"{dataset_name}_{input_length}to{predict_length}_description.txt"), 'w') as f:
            f.write(f'data.shape={data.shape}\n')
            f.write(f"x.shape={x.shape}, y.shape={y.shape}\n")
            f.write(
                f"x_timestamp.shape={x_timestamp.shape}, y.shape={y_timestamp.shape}\n")
            f.write(f"x_tod.shape={x_tod.shape}\n")
            f.write(f"x_dow.shape={x_dow.shape}\n")
            f.write(f'train:\n')
            f.write(
                f"\tx_train.shape={x_train.shape}, y_train.shape={y_train.shape}\n")
            f.write(f"\tx_tod_train.shape={x_tod_train.shape}\n")
            f.write(f"\tx_dow_train.shape={x_dow_train.shape}\n")
            f.write(f'\tmax:{x_train.max()}, min:{x_train.min()}\n')
            f.write(f'\tmean:{x_train.mean()}, std:{x_train.std()}\n')
            train_dt = x_timestamp_train[0, 0].astype(
                datetime.datetime).strftime('%Y-%m-%d %H:%M:%S %A')
            f.write(f'\tstart dt:{train_dt}\n')
            f.write(f'val:\n')
            f.write(
                f"\tx_val.shape={x_val.shape}, y_val.shape={y_val.shape}\n")
            f.write(f"\tx_tod_val.shape={x_tod_val.shape}\n")
            f.write(f"\tx_dow_val.shape={x_dow_val.shape}\n")
            val_dt = x_timestamp_val[0, 0].astype(
                datetime.datetime).strftime('%Y-%m-%d %H:%M:%S %A')
            f.write(f'\tstart dt:{val_dt}\n')
            f.write(f'test:\n')
            f.write(
                f"\tx_test.shape={x_test.shape}, y_test.shape={y_test.shape}\n")
            f.write(f"\tx_tod_test.shape={x_tod_test.shape}\n")
            f.write(f"\tx_dow_test.shape={x_dow_test.shape}\n")
            test_dt = x_timestamp_test[0, 0].astype(
                datetime.datetime).strftime('%Y-%m-%d %H:%M:%S %A')
            f.write(f'\tstart dt:{test_dt}\n')
        print('description file has been generated!')


def data_for_long_term_forecasting(dataset_name, data_file, input_length, predict_length, output_dir,
                                   save=False, save_timestamp_np=False, save_description=False,
                                   train_ratio=0.6, test_ratio=0.2,
                                   slice_size_per_day=288, split_type='int'):
    files = np.load(data_file, allow_pickle=True)
    data = files['data']
    # timestamp=files['timestamp']
    print(data.shape)
    # print(timestamp)
    print("Dataset: ", data.shape, data[5, 0, :])

    # Divide the dataset first ,and construct the sample
    slices = data.shape[0]

    start_datetime = datetime.datetime.strptime(
        start_time_dict[dataset_name], '%Y-%m-%d %H:%M:%S')
    time_delta = datetime.timedelta(minutes=time_interval_dict[dataset_name])
    timestamp_list = []
    dt = start_datetime
    for i in range(slices):
        timestamp_list.append(np.datetime64(dt))
        dt += time_delta
    timestamp_np = np.array(timestamp_list)
    print(timestamp_np.shape)
    if save_timestamp_np:
        np.save(os.path.join(
            output_dir, f"{dataset_name}_timestamp.npy"), timestamp_np)
        print(os.path.join(output_dir,
              f"{dataset_name}_timestamp.npy")+' saved.')

    if timestamp_np is not None:
        start_dt = timestamp_np[0].astype(datetime.datetime)
        time_start_offset = start_dt - \
            datetime.datetime(year=start_dt.year,
                              month=start_dt.month, day=start_dt.day)
        print(timestamp_np[0])
        assert time_start_offset % time_delta == datetime.timedelta(hours=0)
        slice_start = int(time_start_offset / time_delta)
        dayofweek_start = timestamp_np[0].astype(datetime.datetime).weekday()
    else:
        # the start slice id of the day
        slice_start_dict = {'PEMS03': 0, 'PEMS04': 0, 'PEMS07': 0, 'PEMS08': 0,
                            'HZME_INFLOW': 0, 'HZME_OUTFLOW': 0,
                            'T-Drive': 0, 'NYCTaxi': 0, 'CHIBike': 0}
        slice_start = slice_start_dict[dataset_name]
        print('slice_start:', slice_start)
        # the day of the week of the beginning day
        dayofweek_start_dict = {'PEMS03': 5, 'PEMS04': 0, 'PEMS07': 0, 'PEMS08': 4,
                                'HZME_INFLOW': 1, 'HZME_OUTFLOW': 1,
                                'T-Drive': 6, 'NYCTaxi': 2, 'CHIBike': 2}
        dayofweek_start = dayofweek_start_dict[dataset_name]
    print('slice_start:', slice_start)
    print('dayofweek_start:', dayofweek_start)

    tod = [(i + slice_start) % slice_size_per_day for i in range(slices)]
    t_of_d = np.array(tod)
    t_of_d = np.expand_dims(t_of_d, 1)
    print('t_of_d.shape:', t_of_d.shape)

    dow = [(i // slice_size_per_day + dayofweek_start) %
           7 for i in range(slices)]
    d_of_w = np.array(dow)
    d_of_w = np.expand_dims(d_of_w, 1)
    print('d_of_w.shape:', d_of_w.shape)

    if split_type == 'int':
        train_slices = int(slices * train_ratio)
        val_slices = int(slices * test_ratio)
    elif split_type == 'round':
        train_slices = round(slices * train_ratio)
        val_slices = round(slices * test_ratio)
    test_slices = slices - train_slices - val_slices
    train_set = data[: train_slices]
    print(train_set.shape)
    val_set = data[train_slices: val_slices + train_slices]
    print(val_set.shape)
    test_set = data[-test_slices:]
    print(test_set.shape)

    def seq2instance_plus(data, t_of_d, d_of_w, num_his, num_pred):
        num_step = data.shape[0]
        num_sample = num_step - num_his - num_pred + 1
        x = []
        y = []
        x_tod = []
        x_dow = []
        y_tod = []
        y_dow = []
        for i in range(num_sample):
            x.append(data[i: i + num_his])
            y.append(data[i + num_his: i + num_his + num_pred, :, :1])
            x_tod.append(t_of_d[i: i+num_his])
            y_tod.append(t_of_d[i + num_his: i + num_his + num_pred])
            x_dow.append(d_of_w[i: i+num_his])
            y_dow.append(d_of_w[i + num_his: i + num_his + num_pred])

        x = np.array(x)
        y = np.array(y)
        x_tod = np.array(x_tod)
        x_dow = np.array(x_dow)
        y_tod = np.array(y_tod)
        y_dow = np.array(y_dow)
        return x, y, x_tod, x_dow, y_tod, y_dow

    sets = {'train': train_set, 'val': val_set, 'test': test_set}
    tod_sets = {'train': t_of_d[:train_slices],
                'val': t_of_d[train_slices: val_slices + train_slices], 'test': t_of_d[-test_slices:]}
    dow_sets = {'train': d_of_w[:train_slices],
                'val': d_of_w[train_slices: val_slices + train_slices], 'test': d_of_w[-test_slices:]}
    xy = {}
    tod = {}
    dow = {}
    ytod = {}
    ydow = {}
    for set_name in sets.keys():
        data_set = sets[set_name]
        tod_data = tod_sets[set_name]
        dow_data = dow_sets[set_name]
        X, Y, x_tod, x_dow, y_tod, y_dow = seq2instance_plus(data_set[..., :1].astype(
            "float64"), tod_data, dow_data, input_length, predict_length)

        xy[set_name] = [X, Y]
        tod[set_name] = x_tod
        dow[set_name] = x_dow
        ytod[set_name] = y_tod
        ydow[set_name] = y_dow

    x_trains, y_trains = xy['train'][0], xy['train'][1]
    x_vals, y_vals = xy['val'][0], xy['val'][1]
    x_tests, y_tests = xy['test'][0], xy['test'][1]

    x_tod_trains, x_tod_vals, x_tod_tests = tod['train'], tod['val'], tod['test']
    x_dow_trains, x_dow_vals, x_dow_tests = dow['train'], dow['val'], dow['test']
    y_tod_trains, y_tod_vals, y_tod_tests = ytod['train'], ytod['val'], ytod['test']
    y_dow_trains, y_dow_vals, y_dow_tests = ydow['train'], ydow['val'], ydow['test']

    print("train: ", x_trains.shape, y_trains.shape)
    print("val: ", x_vals.shape, y_vals.shape)
    print("test: ", x_tests.shape, y_tests.shape)

    if save:
        output_path = os.path.join(
            output_dir, f"long_term_{dataset_name}_{input_length}to{predict_length}.npz")
        print(f"save file to {output_path}")
        np.savez_compressed(
            output_path,
            train_x=x_trains, train_target=y_trains,
            val_x=x_vals, val_target=y_vals,
            test_x=x_tests, test_target=y_tests,
            train_x_tod=x_tod_trains, val_x_tod=x_tod_vals, test_x_tod=x_tod_tests,
            train_x_dow=x_dow_trains, val_x_dow=x_dow_vals, test_x_dow=x_dow_tests,
            train_y_tod=y_tod_trains, val_y_tod=y_tod_vals, test_y_tod=y_tod_tests,
            train_y_dow=y_dow_trains, val_y_dow=y_dow_vals, test_y_dow=y_dow_tests
        )
        print('npz file has been generated!')

    if save or save_description:
        with open(os.path.join(output_dir, f"long_term_{dataset_name}_{input_length}to{predict_length}_description.txt"), 'w') as f:
            f.write(f'data.shape={data.shape}\n')
            f.write(f"t_of_d.shape={t_of_d.shape}\n")
            f.write(f"d_of_w.shape={d_of_w.shape}\n")
            f.write(f'train:\n')
            f.write(
                f"\tx_trains.shape={x_trains.shape}, y_trains.shape={y_trains.shape}\n")
            f.write(f"\tx_tod_trains.shape={x_tod_trains.shape}\n")
            f.write(f"\tx_dow_trains.shape={x_dow_trains.shape}\n")
            f.write(f'\tmax:{x_trains.max()}, min:{x_trains.min()}\n')
            f.write(f'\tmean:{x_trains.mean()}, std:{x_trains.std()}\n')
            f.write(f'val:\n')
            f.write(
                f"\tx_vals.shape={x_vals.shape}, y_vals.shape={y_vals.shape}\n")
            f.write(f"\tx_tod_val.shape={x_tod_vals.shape}\n")
            f.write(f"\tx_dow_val.shape={x_dow_vals.shape}\n")
            f.write(f'test:\n')
            f.write(
                f"\tx_tests.shape={x_tests.shape}, y_tests.shape={y_tests.shape}\n")
            f.write(f"\tx_tod_tests.shape={x_tod_tests.shape}\n")
            f.write(f"\tx_dow_tests.shape={x_dow_tests.shape}\n")
        print('description file has been generated!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='none')
    # parser.add_argument("--dataset_name", type=str, default="PEMS08", help="Dataset name.")
    # parser.add_argument("--input_length", type=int,
    #                     default=12, help="Sequence Length.",)
    # parser.add_argument("--predict_length", type=int,
    #                     default=12, help="Sequence Length.",)
    # parser.add_argument("--train_ratio", type=float,
    #                     default=0.6, help="train ratio", )
    # parser.add_argument("--test_ratio", type=float,
    #                     default=0.2, help="test ratio", )
    parser.add_argument("--save", type=int, default=0, help="save data", )
    parser.add_argument("--save_timestamp_np", type=int,
                        default=0, help="save_timestamp_np", )
    parser.add_argument("--save_description", type=int,
                        default=0, help="save_description_txt", )
    parser.add_argument("--split_type", type=str, default="round")
    args = parser.parse_args()


    if os.path.exists(f'./configurations/{args.config_file}'):
        with open(f'./configurations/{args.config_file}', 'r') as f:
            x = json.load(f)
            data_config = x['data_config']
            args.dataset_name = data_config['dataset_name']
            args.input_length = data_config['input_length']
            args.predict_length = data_config['predict_length']
            args.train_ratio = data_config['train_ratio']
            args.test_ratio = data_config['test_ratio']
            args.split_type = data_config['split_type']
            slice_size_per_day = data_config['slice_size_per_day']
    else:
        print('ERROR: no configuration')

    
    output_dir = f"./datasets/{args.dataset_name}"
    if args.save == 1 and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.input_length <= 12:
        filetype_dict = {'PEMS04': 'npz',
                         'PEMS07': 'npz',
                         'PEMS08': 'npz',
                         'T-Drive': 'grid',
                         'NYCTaxi': 'grid',
                         'CHIBike': 'grid'}
        data_file = f"./datasets/{args.dataset_name}.{filetype_dict[args.dataset_name]}"
        # For fair comparison, this function is in line with the data preparation method in PDFormer
        data_for_short_term_forecasting(args.dataset_name, data_file, args.input_length, args.predict_length, output_dir,
                                        save=bool(args.save), save_timestamp_np=bool(args.save_timestamp_np),
                                        save_description=bool(
                                            args.save_description),
                                        train_ratio=args.train_ratio, slice_size_per_day=slice_size_per_day,
                                        split_type=args.split_type)
    else:
        filename_dict = {
            'PEMS04': 'pems04_1dim.npz',
            'PEMS08': 'pems08_1dim.npz'
        }
        data_file = f"./datasets/{filename_dict[args.dataset_name]}"
        
        # For fair comparison, this function is in line with the data preparation method in SSTBAN
        data_for_long_term_forecasting(args.dataset_name, data_file, args.input_length, args.predict_length, output_dir,
                                       save=bool(args.save), save_timestamp_np=bool(args.save_timestamp_np),
                                       save_description=bool(
                                           args.save_description),
                                       train_ratio=args.train_ratio, slice_size_per_day=slice_size_per_day,
                                       split_type=args.split_type)
