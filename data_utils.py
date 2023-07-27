import numpy as np
import os
from logging import getLogger

class MinMaxScaler():
    """
    Standard the input
    """

    def __init__(self, max, min):
        self.max = max
        self.min = min

        print('_max.shape:', max.shape)
        print('_min.shape:', min.shape)


    def transform(self, data):
        x=1.*(data - self.min) / (self.max - self.min)
        return 2. *x-1.

    def inverse_transform(self, data):
        x= (data+1.)/2.
        x=x*(self.max - self.min) + self.min
        return x

class NoneScaler():
    """
    Standard the input
    """

    def __init__(self, type='none'):
        self.type=type

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class DataLoader_time(object):
    def __init__(self, x, y, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        xs, xtod, xdow = x
        ys = y

        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            xtod_padding = np.repeat(xtod[-1:], num_padding, axis=0)
            xdow_padding = np.repeat(xdow[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            xtod = np.concatenate([xtod, xtod_padding], axis=0)
            xdow = np.concatenate([xdow, xdow_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.xtod = xtod
        self.xdow = xdow
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        xtod = self.xtod[permutation]
        xdow = self.xdow[permutation]
        self.xs = xs
        self.xtod = xtod
        self.xdow = xdow
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                xtod_i = self.xtod[start_ind: end_ind, ...]
                xdow_i = self.xdow[start_ind: end_ind, ...]

                y_i = self.ys[start_ind: end_ind, ...]
                yield ([x_i, xtod_i, xdow_i], y_i)
                self.current_ind += 1

        return _wrapper()


def load_dataset_time(dataset_name, input_length, predict_length, batch_size, valid_batch_size=None, test_batch_size=None,
scalertype='zscore', in_dim=1):
    data = {}
    logger = getLogger()
    dataset_dir = f'./datasets/{dataset_name}'
    tip=f'{input_length}to{predict_length}'    
    print('tip in dataloader:',tip)
    
    if input_length>20:
        cat_data = np.load(os.path.join(dataset_dir, 'long_term_' + dataset_name + f'_{tip}.npz'))
        num_train_samples = cat_data['train_x'].shape[0]
        num_val_samples = cat_data['val_x'].shape[0]
        num_test_samples = cat_data['test_x'].shape[0]
        set_train_sample_size = {'0':num_train_samples,'1':num_val_samples*2,'2':num_val_samples}['0']
        set_sample_size={'train':set_train_sample_size,'val':num_val_samples,'test':num_test_samples}
        for category in ['train', 'val','test']:
            sample_size=set_sample_size[category]
            data['x_' + category] = cat_data[category+'_x'][-sample_size:,...]
            data['y_' + category] = cat_data[category+'_target'][-sample_size:,...]
            data['x_tod_' + category] = cat_data[category+'_x_tod'][-sample_size:,...]
            data['x_dow_' + category] = cat_data[category+'_x_dow'][-sample_size:,...]
    else:
        cat_data = np.load(os.path.join(dataset_dir, dataset_name + f'_{tip}.npz'))
        for category in ['train', 'val','test']:
            data['x_' + category] = cat_data['x_' + category]
            data['y_' + category] = cat_data['y_' + category]
            data['x_tod_' + category] = cat_data['x_tod_' + category]
            data['x_dow_' + category] = cat_data['x_dow_' + category]

    logger.info("data['x_train'].shape="+str(data['x_train'].shape))
    
    assert scalertype in ['minmax','zscore','none']

    x_train = data['x_train']
    if scalertype=='minmax':
        _max = x_train[...,:in_dim] .max()
        _min = x_train[...,:in_dim] .min()
        scaler = MinMaxScaler(max=_max,min=_min)
    elif scalertype=='zscore':
        scaler = StandardScaler(mean=x_train[...,:in_dim] .mean(), std=x_train[...,:in_dim] .std())
    elif scalertype=='none':
        scaler = NoneScaler()
    
    for category in ['train', 'val', 'test']:
        data['x_' + category][...,:in_dim]  = scaler.transform(data['x_' + category][...,:in_dim] )
   
    data['train_loader'] = DataLoader_time([data['x_train'], data['x_tod_train'], data['x_dow_train']],
                                      data['y_train'], batch_size)
    data['val_loader'] = DataLoader_time([data['x_val'], data['x_tod_val'], data['x_dow_val']],
                                     data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader_time([data['x_test'], data['x_tod_test'], data['x_dow_test']], 
                                    data['y_test'], test_batch_size)

    data['scaler'] = scaler
    logger.info('train_x.shape:'+str(data['x_train'].shape))
    logger.info('val_x.shape:'+str(data['x_val'].shape))
    logger.info('test_x.shape:'+str(data['x_test'].shape))

    num_samples = data['x_train'].shape[0]
    seq_len = data['x_train'].shape[1]
    num_nodes = data['x_train'].shape[2]
    in_dim = data['x_train'].shape[-1]
    logger.info('num_nodes:'+str(num_nodes))

    return data, num_nodes, in_dim

