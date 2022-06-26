__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

import torch
from torchvision import transforms
import numpy as np
import time
import glob
import random
import argparse
import h5py
import json
import torch.nn.init as init
import pandas as pd 

from config import  *
from sys_utils import *
from vsum_tools import  *
from vasnet_model import  *


def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

def parse_splits_filename(splits_filename):
    #* ex. input = 'splits/tvsum_splits.json'
    # Parse split file and count number of k_folds
    spath, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    dataset_name = sfname.split('_')[0]  # Get dataset name e.g. tvsum

    # Get number of discrete splits within each split json file
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    return dataset_name, splits

def lookup_weights_splits_file(path, dataset_name, dataset_type, split_id):
    dataset_type_str = '' if dataset_type == '' else dataset_type + '_'
    weights_filename = path + '/models/{}_{}splits_{}_*.tar.pth'.format(dataset_name, dataset_type_str, split_id)
    weights_filename = glob.glob(weights_filename)
    if len(weights_filename) == 0:
        print("Couldn't find model weights: ", weights_filename)
        return ''

    # Get the first weights file in the dir
    weights_filename = weights_filename[0]
    splits_file = path + '/splits/{}_{}splits.json'.format(dataset_name, dataset_type_str)

    return weights_filename, splits_file


class AONet:

    def __init__(self, hps: HParameters):
        self.hps = hps
        self.model = None
        self.log_file = None
        self.verbose = hps.verbose


    def fix_keys(self, keys, dataset_name = None):
        """
        :param keys:
        :return:
        """
        # dataset_name = None
        if len(self.datasets) == 1:
            dataset_name = next(iter(self.datasets))
            # print(f'\n L#83 : dataset_name : {dataset_name}')

        keys_out = []
        for key in keys:
            t = key.split('/')
            if len(t) != 2:
                assert dataset_name is not None, "ERROR dataset name in some keys is missing but there are multiple dataset {} to choose from".format(len(self.datasets))

                key_name = dataset_name+'/'+key
                keys_out.append(key_name)
            else:
                keys_out.append(key)
        # print(f'keys_out : {keys_out}')
        return keys_out


    def load_datasets(self, dataset = None):
        """
        Loads all h5 datasets from the datasets list into a dictionary self.dataset
        referenced by their base filename
        :param datasets:  List of dataset filenames
            (e.g.) ["vip_summe_inorder_length_25_9000", "vip_summe_reversed_length_25_9000"]
        :return:
        """
        if dataset is None:
            dataset = self.hps.datasets

        dir_path = [
            '/data/project/rw/video_summarization/dataset/exp1_Order/',
            '/data/project/rw/video_summarization/dataset/exp2_ConcatRatio_and_Type/',
            '/data/project/rw/video_summarization/dataset/exp3_VideoLength/'
        ]

        if self.hps.expr == "exp1":
            base_dir = dir_path[0]
        elif self.hps.expr == "exp2":
            base_dir = dir_path[1]
        elif self.hps.expr == "exp3":
            base_dir = dir_path[2]
        else:
            raise NotImplementedError("only implemented exp1, exp2, exp3")

        datasets_dict = {}
        # for dataset in dataset:
        datasets_dict[dataset] = h5py.File(f'{base_dir}/{dataset}.h5', 'r')
        self.datasets = datasets_dict
        return datasets_dict


    def load_split_file(self, splits_file):

        self.dataset_name, self.splits = parse_splits_filename(splits_file)
        self.split_file = splits_file
        print("Loading splits from: ",splits_file)

        # return n_folds


    def select_split(self, idx):
        split = self.splits
        self.train_keys = split['train_keys']
        self.test_keys = split['test_keys']

        dataset_filename = self.hps.get_dataset_by_name(self.dataset_name)[idx]
        # print(f'dataset_filename : {dataset_filename}')
        # _,dataset_filename = os.path.split(dataset_filename)
        # dataset_filename,_ = os.path.splitext(dataset_filename)
        self.train_keys = self.fix_keys(self.train_keys, dataset_filename) 
            #* "video_1" -> "vip_summe_inorder_length_25_9000/video_1" 이런식으로 변함
        self.test_keys = self.fix_keys(self.test_keys, dataset_filename)
        return



    def load_model(self, model_filename):
        self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        return


    def initialize(self, cuda_device=None):
        rnd_seed = 12345
        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        torch.manual_seed(rnd_seed)

        self.model = VASNet()
        self.model.eval()
        self.model.apply(weights_init)
        #print(self.model)

        cuda_device = cuda_device or self.hps.cuda_device

        if self.hps.use_cuda:
            print("Setting CUDA device: ",cuda_device)
            torch.cuda.set_device(cuda_device)
            torch.cuda.manual_seed(rnd_seed)

        if self.hps.use_cuda:
            self.model.cuda()

        return


    def get_data(self, key):
        key_parts = key.split('/')
        assert len(key_parts) == 2, "ERROR. Wrong key name: "+key
        dataset, key = key_parts
        # print(f'dataset, key : {dataset, key}')
        # print(f'self.datasets[dataset][key] : {self.datasets[dataset][key]}')
        return self.datasets[dataset][key]

    def lookup_weights_file(self, data_path):
        """
        weights_filename -> 
        exp1/summe/vip_summe_inorder_length_25_9000
        /models/4_0.3402816259999202.tar.pth
        """
        # dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
        weights_filename = data_path + '/models/*.tar.pth'
        weights_filename = glob.glob(weights_filename)
        if len(weights_filename) == 0:
            print("Couldn't find model weights: ", weights_filename)
            return ''

        # Get the first weights filename in the dir
        weights_filename = weights_filename[0]
        return weights_filename


    def train(self, output_dir='EX-0'):

        print("Initializing VASNet model and optimizer...")
        self.model.train()

        criterion = nn.MSELoss()

        if self.hps.use_cuda:
            criterion = criterion.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=self.hps.lr[0], weight_decay=self.hps.l2_req)

        print("Starting training...")

        max_val_fscore = 0
        max_val_fscore_epoch = 0
        train_keys = self.train_keys[:]

        lr = self.hps.lr[0]
        for epoch in range(self.hps.epochs_max):

            print("Epoch: {0:6}".format(str(epoch+1)+"/"+str(self.hps.epochs_max)), end='')
            self.model.train()
            avg_loss = []

            random.shuffle(train_keys)

            for i, key in enumerate(train_keys):
                dataset = self.get_data(key) #self.datasets[dataset][key]
                seq = dataset['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                target = dataset['gtscore'][...]
                target = torch.from_numpy(target).unsqueeze(0)

                # Normalize frame scores
                target -= target.min()
                target /= target.max()

                if self.hps.use_cuda:
                    seq, target = seq.float().cuda(), target.float().cuda()

                seq_len = seq.shape[1]
                y, _ = self.model(seq,seq_len)
                # loss_att = 0

                loss = criterion(y, target)
                # loss2 = y.sum()/seq_len
                # loss = loss + loss_att
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # avg_loss.append([float(loss), float(loss_att)])
                avg_loss.append(float(loss))


            # Evaluate test dataset
            val_fscore, video_scores = self.eval(self.test_keys)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch

            avg_loss = np.array(avg_loss)
            print("   Train loss: {0:.05f}".format(np.mean(avg_loss)), end='')
            print('   Test F-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore, max_val_fscore))

            if self.verbose:
                video_scores = [["No", "Video", "F-score"]] + video_scores
                print_table(video_scores, cell_width=[3,40,8])

            # Save model weights
            path, filename = os.path.split(self.split_file) # 'splits', 'tvsum_splits.json'
            base_filename, _ = os.path.splitext(filename) # tvsum_splits
            path = os.path.join(output_dir, 'models_temp')
            os.makedirs(path, exist_ok=True)
            filename = str(epoch)+'_' + str(round(val_fscore*100,3))+'.pth.tar'
            torch.save(self.model.state_dict(), os.path.join(path, filename))

        return max_val_fscore, max_val_fscore_epoch


    def eval(self, keys, results_filename=None, call='train'):

        self.model.eval()
        summary = {}
        att_vecs = {}
        with torch.no_grad():
            for i, key in enumerate(keys):
                data = self.get_data(key)
                # seq = self.dataset[key]['features'][...]
                seq = data['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)

                if self.hps.use_cuda:
                    seq = seq.float().cuda()

                y, att_vec = self.model(seq, seq.shape[1])
                summary[key] = y[0].detach().cpu().numpy()
                att_vecs[key] = att_vec.detach().cpu().numpy()

        if call == 'train':
            f_score, video_scores = self.eval_summary(summary, keys, metric=self.dataset_name,
                results_filename=results_filename, att_vecs=att_vecs, call=call)
            return f_score, video_scores
        
        else:
            f_score, video_scores, df = self.eval_summary(summary, keys, metric=self.dataset_name,
                results_filename=results_filename, att_vecs=att_vecs, call=call)
            return f_score, video_scores, df


    def eval_summary(self, machine_summary_activations, test_keys, results_filename=None, metric='tvsum', att_vecs=None, call='train'):

        eval_metric = 'avg' if metric == 'tvsum' else 'max'

        if results_filename is not None:
            h5_res = h5py.File(results_filename, 'w')
        
        if call == 'eval':
            df = pd.DataFrame()

        fms = []
        video_scores = []
        for key_idx, key in enumerate(test_keys):
            video_name = key.split('/')[-1]
            f_scores = []
            d = self.get_data(key)
            probs = machine_summary_activations[key]

            if 'change_points' not in d:
                print("ERROR: No change points in dataset/video ",key)

            cps = d['change_points'][...]
            num_frames = d['n_frames'][()]
            nfps = d['n_frame_per_seg'][...].tolist()
            positions = d['picks'][...]
            user_summary = d['user_summary'][...]
            sum_ratio = d['sum_ratio'][...]
            video_boundary = d['video_boundary'][...]

            for user_id in range(user_summary.shape[0]):
                machine_summary = generate_summary(probs, cps, num_frames, nfps, positions, proportion=sum_ratio[user_id])
                fm, _, _ = evaluate_summary(machine_summary, user_summary[user_id], eval_metric)
                f_scores.append(fm)
                if call == 'eval':
                    coverage = coverage_count(video_name, user_id, machine_summary, user_summary[user_id], video_boundary, sum_ratio[user_id])
                    df = df.append(coverage, ignore_index=True)

            if eval_metric == 'avg':
                final_f_score = np.mean(f_scores)
            elif eval_metric == 'max':
                final_f_score = np.max(f_scores)
            fms.append(final_f_score)

            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(final_f_score)])

            if results_filename:
                gt = d['gtscore'][...]
                h5_res.create_dataset(key + '/score', data=probs)
                h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                h5_res.create_dataset(key + '/gtscore', data=gt)
                h5_res.create_dataset(key + '/fm', data=fm)
                h5_res.create_dataset(key + '/picks', data=positions)

                video_name = key.split('/')[1]
                if 'video_name' in d:
                    video_name = d['video_name'][...]
                h5_res.create_dataset(key + '/video_name', data=video_name)

                if att_vecs is not None:
                    h5_res.create_dataset(key + '/att', data=att_vecs[key])

        mean_fm = np.mean(fms)

        # Reporting & logging
        if results_filename is not None:
            h5_res.close()

        if call == 'train':
            return mean_fm, video_scores
        
        else:
            df = df[['video_id', 'user_id', 'sum_ratio', 'v1_frames', 'v1_pred_frames', 'v1_gt_frames', 'v1_n_overlap', 'v1_overlap_ratio', 'v1_pred_sum_ratio', 'v1_gt_sum_ratio',\
            'v2_frames', 'v2_pred_frames', 'v2_gt_frames', 'v2_n_overlap', 'v2_overlap_ratio', 'v2_pred_sum_ratio', 'v2_gt_sum_ratio',\
            'v3_frames', 'v3_pred_frames', 'v3_gt_frames', 'v3_n_overlap', 'v3_overlap_ratio', 'v3_pred_sum_ratio', 'v3_gt_sum_ratio',\
            'v4_frames', 'v4_pred_frames', 'v4_gt_frames', 'v4_n_overlap', 'v4_overlap_ratio', 'v4_pred_sum_ratio', 'v4_gt_sum_ratio']]
            
            return mean_fm, video_scores, df




#==============================================================================================



def eval_split(hps, splits_filename, datasets, data_dir):
    """
    splits_filename : 'splits/tvsum_splits.json' or 'splits/summe_splits.json'
    """

    print("\n")

    val_f_scores = []

    for idx, expr_data in enumerate(datasets):
        ''' expr_data => 같은 데이터셋 안에(summe / tvsum) 다른 h5 file들 '''
        
        ao = AONet(hps)
        ao.initialize()
        ao.load_datasets(dataset=expr_data)
        ao.load_split_file(splits_file=splits_filename)
        ao.select_split(idx)

        weights_file_path = data_dir + '/' + expr_data
        weights_filename = ao.lookup_weights_file(weights_file_path)
        # exp1/summe/vip_summe_inorder_length_25_9000
        print("Loading model:", weights_filename)
        ao.load_model(weights_filename)
        val_fscore, video_scores, df = ao.eval(ao.test_keys, call='eval')
        df.to_csv(f'{weights_file_path}/best_epoch_results.csv', index=False)
        val_f_scores.append([expr_data, val_fscore])

        if hps.verbose:
            video_scores = [["No.", "Video", "F-score"]] + video_scores
            print_table(video_scores, cell_width=[4,45,5])

        # print("Avg F-score: ", val_fscore)
        # print("")

    return val_f_scores


def train(hps):
    os.makedirs(hps.output_dir, exist_ok=True)
    # os.makedirs(os.path.join(hps.output_dir, 'splits'), exist_ok=True)
    # os.makedirs(os.path.join(hps.output_dir, 'code'), exist_ok=True)
    # os.makedirs(os.path.join(hps.output_dir, 'models'), exist_ok=True)
    # os.system('cp -f splits/*.json  ' + hps.output_dir + '/splits/')
    # os.system('cp *.py ' + hps.output_dir + '/code/')

    # Create a file to collect results from all splits

    for split_filename in hps.splits:
        '''
        hps.splits => test_key, train_key있는 json file임. 
        self.splits = ['splits/tvsum_splits.json',
                        'splits/summe_splits.json']
        self.splits += ['splits/tvsum_aug_splits.json',
                        'splits/summe_aug_splits.json']
        '''

        dataset_name, splits = parse_splits_filename(split_filename)

        # For no augmentation use only a dataset corresponding to the split file
        # if dataset_type == '':
        datasets = hps.get_dataset_by_name(dataset_name) # 처음에는 summe h5 파일들만...
        
        # Create a file to collect results from all splits
        os.makedirs(hps.output_dir + dataset_name, exist_ok=True)
        f = open(f'{hps.output_dir}{dataset_name}/results.txt', 'wt')
        for idx, expr_data in enumerate(datasets):
            # if datasets is None:
                # datasets = hps.datasets
            print(f'expr_data: {expr_data}')

            ao = AONet(hps)
            ao.initialize()
            ao.load_datasets(dataset=expr_data)
            ao.load_split_file(splits_file=split_filename)
            ao.select_split(idx)

            fscore, fscore_epoch = ao.train(output_dir=f'{hps.output_dir}/{dataset_name}/{expr_data}')

            # Log F-score for this split
            f.write(expr_data + ', ' + str(fscore) + ', ' + str(fscore_epoch) + '\n')
            f.flush()

            # Save model with the highest F score
            _, log_file = os.path.split(split_filename) #tvsum_splits.json
            log_dir, _ = os.path.splitext(log_file) #tvsum_splits
            log_file = os.path.join(hps.output_dir, dataset_name, expr_data, 'models/') + str(fscore_epoch) + '_' + str(fscore) + '.tar.pth'

            os.makedirs(os.path.join(hps.output_dir, dataset_name, expr_data, 'models'), exist_ok=True)
            os.system('cp ' + hps.output_dir + dataset_name + '/' + expr_data + '/models_temp/' + str(fscore_epoch) + '_*.pth.tar ' + log_file)
            # os.system('rm -rf ' + hps.output_dir + '/models_temp/')

            print("Split: {0:}   Best F-score: {1:0.5f}   Model: {2:}".format(split_filename, fscore, log_file))

        # Write average F-score for all splits to the results.txt file
        # f.write(split_filename + ', ' + str('avg') + ', ' + str(f_avg) + '\n')
        # f.flush()

    f.close()


if __name__ == "__main__":
    print_pkg_versions()

    parser = argparse.ArgumentParser("PyTorch implementation of paper \"Summarizing Videos with Attention\"")
    parser.add_argument('-r', '--root', type=str, default='', help="Project root directory")
    parser.add_argument('-d', '--datasets', type=str, help="Path to a comma separated list of h5 datasets")
    parser.add_argument('-s', '--splits', type=str, help="Comma separated list of split files.")
    parser.add_argument('-t', '--train', action='store_true', help="Train")
    parser.add_argument('-v', '--verbose', action='store_true', help="Prints out more messages")
    # parser.add_argument('-o', '--output_dir', type=str, required=True, help="output dir. name")
    parser.add_argument('--expr', type=str, required=True, choices=['exp1', 'exp2', 'exp3'], help="Experiment name")
    args = parser.parse_args()

    # MAIN
    #======================
    hps = HParameters()
    hps.load_from_args(args.__dict__)

    print("Parameters:")
    print("----------------------------------------------------------------------")
    print(hps)

    if hps.train:
        train(hps)
    else:
        for split_filename in hps.splits: # json files
            results=[['No', 'Dataset', 'Expr Data' , 'Best F-score']]
            dataset_name, splits = parse_splits_filename(split_filename)
            datasets = hps.get_dataset_by_name(dataset_name)

            f_score = eval_split(hps, split_filename, datasets = datasets, data_dir = hps.output_dir + dataset_name)
            for i, fs in enumerate(f_score):
                results.append([i+1, dataset_name, fs[0], str(round(fs[1] * 100.0, 3))+"%"])

            print(f"\nFinal Results({dataset_name}):")
            print_table(results)


    sys.exit(0)

