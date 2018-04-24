import numpy as np

class DataProvider(object):
    def __init__(self, data_path='data.npy', test_data='test.npy', batch_size=16):
        self.data = np.load(data_path)
        self.batch_size = batch_size
        self.font_num = self.data.shape[0]
        self.charactor_num = self.data.shape[1]
        self.test_data = np.load(test_data)

    def get_batch(self):
        data = []
        label = []
        for i in range(self.batch_size):
            style_src = int(np.random.rand()* self.font_num)
            charactor_num = int(np.random.rand()* self.charactor_num)
            for j in range(3):
                random_charactor = int(np.random.rand()* self.charactor_num)
                if j==0:
                    tmp_data = self.data[style_src][random_charactor].reshape([256,256,1])
                    tmp_data = np.concatenate([tmp_data, tmp_data, tmp_data], axis=2)
                else:
                    tmp = self.data[style_src][random_charactor].reshape([256,256,1])
                    for k in range(3):
                        tmp_data = np.concatenate([tmp_data, tmp], axis=2)
            tmp = self.data[style_src][charactor_num].reshape([256,256,1])
            for k in range(3):
                tmp_data = np.concatenate([tmp_data, tmp], axis=2)
            data.append(tmp_data)
            label.append(charactor_num)
        data = np.array(data, dtype=np.float32)/127.5 - 1
        label = np.array(label)
        return data, label

    def get_test(self):
        data = []
        label = []
        for i in range(16):
            for j in range(3):
                random_charactor = int(np.random.rand()* 50)
                if j==0:
                    tmp_data = self.test_data[random_charactor].reshape([256,256,1])
                    tmp_data = np.concatenate([tmp_data, tmp_data, tmp_data], axis=2)
                else:
                    tmp = self.test_data[random_charactor].reshape([256,256,1])
                    for k in range(3):
                        tmp_data = np.concatenate([tmp_data, tmp], axis=2)
            tmp = self.test_data[i].reshape([256,256,1])
            for k in range(3):
                tmp_data = np.concatenate([tmp_data, tmp], axis=2)
            data.append(tmp_data)
            label.append(i)
        data = np.array(data, dtype=np.float32)/127.5 - 1
        label = np.array(label)
        return data, label
