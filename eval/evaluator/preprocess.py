import os
import requests
import inspect
import random
from utils import *


class BBTFinCUGE:

    def __init__(self):
        # 原始数据路径
        self.raw_path = 'data/fincuge/raw'
        # 处理后的数据路径
        self.saved_path = 'data/fincuge'
        # 超对称数据集列表
        self.eval_datasets = ['fincqa', 'finese', 'finfe', 'finna', 'finnl', 'finnsp', 'finqa', 'finre']
        # 训练集与测试集
        self.splits = ['train', 'eval']
        # 数据链接
        self.url = 'https://raw.githubusercontent.com/ssymmetry/BBT-FinCUGE-Applications/main/FinCUGE_Publish/{dataset}/{split}_list.json'
        # 创建文件夹
        os.makedirs(self.raw_path, exist_ok=True)

    def download_all(self):
        # 下载所有数据集
        for eval_dataset in self.eval_datasets:
            try:
                for split in self.splits:
                    file_path = os.path.join(self.raw_path, '{}-{}.json'.format(eval_dataset, split))
                    if not os.path.exists(file_path):
                        file_url = self.url.format(dataset=eval_dataset, split=split)
                        print('download {}-{} dataset from {}'.format(eval_dataset, split, file_url))
                        response = requests.get(file_url)
                        if response.status_code == 200:
                            data = json.loads(response.content)
                            write_json(file_path, data)
                            print('successful to download dataset {}, saved to {}'.format(eval_dataset, file_path))
                        else:
                            print('failed to download dataset {}, status_code: {}'.format(eval_dataset, response.status_code))
                    else:
                        print('found existing {}-{}, skipping download ...'.format(eval_dataset, split))
            except Exception as e:
                print('failed to download dataset {}, {}'.format(eval_dataset, e))

    def process_finfe(self, split: str):
        eval_dataset = 'finfe'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        emotion_text_maps = {0: '消极', 1: '中性', 2: '积极'}
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances = []
        for i, example in enumerate(data):
            instance = {
                "id": '-'.join([split, str(i + 1)]),
                "input": example[0],
                "gold_answer": emotion_text_maps.get(example[1]),
                'source': eval_dataset
            }
            instances.append(instance)
        return instances

    def process_finqa(self, split: str):
        eval_dataset = 'finqa'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances = []
        for i, example in enumerate(data):
            instance = {
                "id": '-'.join([split, str(i + 1)]),
                "input": example[0],
                "gold_answer": example[1],
                'source': eval_dataset
            }
            instances.append(instance)
        return instances

    def process_fincqa(self, split: str):
        eval_dataset = 'fincqa'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances = []
        for i, example in enumerate(data):
            instance = {
                "id": '-'.join([split, str(i + 1)]),
                "input": example[0],
                "gold_answer": example[1],
                'source': eval_dataset
            }
            instances.append(instance)
        return instances

    def process_finna(self, split: str):
        eval_dataset = 'finna'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances = []
        for i, example in enumerate(data):
            instance = {
                "id": '-'.join([split, str(i + 1)]),
                "input": example[1],
                "gold_answer": example[0],
                'source': eval_dataset
            }
            instances.append(instance)
        return instances

    def process_finre(self, split: str):
        eval_dataset = 'finre'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances = []
        for i, example in enumerate(data):
            instance = {
                "id": '-'.join([split, str(i + 1)]),
                "input": [example[3], example[0], example[1]],
                "gold_answer": example[2],
                'source': eval_dataset
            }
            instances.append(instance)
        return instances

    def process_finnl(self, split: str):
        eval_dataset = 'finnl'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances = []
        for i, example in enumerate(data):
            instance = {
                "id": '-'.join([split, str(i + 1)]),
                "input": example[0],
                "gold_answer": example[1],
                'source': eval_dataset
            }
            instances.append(instance)
        return instances

    def process_finnsp(self, split: str):
        eval_dataset = 'finnsp'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances1, instances2 = [], []
        for i, example in enumerate(data):
            instance1 = {
                "id": '-'.join([split, str(i + 1)]),
                "input": example[2],
                "gold_answer": example[4],
                'source': eval_dataset
            }
            instances1.append(instance1)
        data2 = []
        for list in data:
            if list[4]=='1':
                data2.append(list)
        for i, example in enumerate(data2):
            instance2 = {
                "id": '-'.join([split, str(i + 1)]),
                "input": [example[2], example[3]],
                "gold_answer": example[5],
                'source': eval_dataset
            }
            instances2.append(instance2)
        return instances1, instances2

    def process_finese(self, split: str):
        eval_dataset = 'finese'
        # 获取函数名
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        # 确保函数名与数据集名称相对应
        assert eval_dataset == func_name.replace('process_', '')
        data = load_json(os.path.join(self.raw_path, '{}-{}.json').format(eval_dataset, split))
        instances = []
        for i, example in enumerate(data):
            instance = {
                "id": '-'.join([split, str(i + 1)]),
                "input": [example[1], example[2]],
                "gold_answer": example[3],
                'source': eval_dataset
            }
            instances.append(instance)
        return instances

    # 定义每个数据集的数据处理函数 ...
    # ['fincqa', 'finese', 'finfe', 'finna', 'finne', 'finnl', 'finnsp', 'finqa', 'finre']
    # def process_fincqa(self, split: str):
    #     pass
    #
    # def process_finese(self, split: str):
    #     pass
    #
    # def process_finna(self, split: str):
    #     pass


if __name__ == '__main__':

    random.seed(123)

    process = BBTFinCUGE()
    process.download_all()

    # 每个数据集随机抽取10个样本
    num_instances = 10
    instruct_samples = {}

    for dataset in process.eval_datasets:
        # if dataset == 'finre':
        # 如果有一些数据集需要处理成多个数据集，则使用if来进行分支
        #     pass
        # else:
        #     pass
        if dataset != 'finnsp':
            process_func = getattr(process, 'process_' + dataset, None)
            if process_func is not None:
                # 对于训练集，随机挑选k个样本
                train_instances = process_func(split='train')
                instruct_samples[dataset] = random.sample(train_instances, k=num_instances)
                # 对于测试集，保存到对应路径
                eval_instances = process_func(split='eval')
                write_json(os.path.join(process.saved_path, dataset + '-eval.jsonl'), eval_instances)
        else:
            process_func = getattr(process, 'process_' + dataset, None)
            if process_func is not None:
                # 对于训练集，随机挑选k个样本
                train_instances1, train_instances2 = process_func(split='train')
                instruct_samples[dataset + '1'] = random.sample(train_instances1, k=num_instances)
                instruct_samples[dataset + '2'] = random.sample(train_instances2, k=num_instances)
                # 对于测试集，保存到对应路径
                eval_instances1, eval_instances2 = process_func(split='eval')
                write_json(os.path.join(process.saved_path, dataset + '1' + '-eval.jsonl'), eval_instances1)
                write_json(os.path.join(process.saved_path, dataset + '2' + '-eval.jsonl'), eval_instances2)

    write_json(os.path.join(process.saved_path, 'instruct_samples.json'), instruct_samples)
