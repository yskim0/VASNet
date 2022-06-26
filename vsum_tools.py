''''
Courtesy of KaiyangZhou
https://github.com/KaiyangZhou/pytorch-vsumm-reinforce

@article{zhou2017reinforcevsumm,
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao},
   journal={arXiv:1801.00054},
   year={2017}
}

Modifications by Jiri Fajtl
- knapsack replaced with knapsack_ortools
- added evaluate_user_summaries() for user summaries ground truth evaluation
'''

import numpy as np
#from knapsack import knapsack_dp
from knapsack import knapsack_ortools
import math


def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        #picks = knapsack_dp(seg_score, nfps, n_segs, limits)
        picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    return summary


def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_frames = len(user_summary)

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    overlap_duration = (machine_summary * user_summary).sum()
    precision = overlap_duration / (machine_summary.sum() + 1e-8)
    recall = overlap_duration / (user_summary.sum() + 1e-8)
    if precision == 0 and recall == 0:
        f_score = 0.
    else:
        f_score = (2 * precision * recall) / (precision + recall)

    
    return f_score, precision, recall

    
def coverage_count(vid, uid, predicted_summary, user_summary, video_boundary, sum_ratio):
    """ #TODO
    :param ndarray predicted_summary: The generated summary from our model.
    :param ndarray user_summary: The user defined ground truth summaries (or summary).
    :return: The reduced fscore based on the eval_method
    """
    max_len = max(len(predicted_summary), len(user_summary))
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    G[:len(user_summary)] = user_summary

    G_split = np.split(G, video_boundary + 1)[:-1] # last element is empty
    S_split = np.split(S, video_boundary + 1)[:-1] # last element is empty 

    # for dataframe
    raw_data = {}
    raw_data['video_id'] = vid
    raw_data['user_id'] = int(uid)
    raw_data['sum_ratio'] = sum_ratio

    for i, g_seg in enumerate(G_split):
        s_seg = S_split[i]
        # print(f'len(s_seg) : {len(s_seg)}')
        # print(f'len(g_seg) : {len(g_seg)}')

        n_pred_s_frame = np.count_nonzero(s_seg) # number of summary frames of predicted summary
        n_gt_s_frame = np.count_nonzero(g_seg)
        seg_sum_ratio = n_pred_s_frame / len(g_seg)
        overlapped = s_seg & g_seg
        n_overlapped = np.count_nonzero(overlapped)

        raw_data[f'v{i+1}_frames'] = len(g_seg) # 해당 비디오 세그먼트의 총 프레임 수
        raw_data[f'v{i+1}_pred_frames'] = n_pred_s_frame # 해당 세그먼트 내에서 machine summary가 sumamry라고 예측한 프레임 개수
        raw_data[f'v{i+1}_gt_frames'] = n_gt_s_frame # 해당 세그먼트 내에서 gt summary가 sumamry라고 예측한 프레임 개수
        raw_data[f'v{i+1}_n_overlap'] = n_overlapped
        raw_data[f'v{i+1}_overlap_ratio'] = n_overlapped / (n_gt_s_frame + 1e-6) # to avoid divide by zero error
        raw_data[f'v{i+1}_pred_sum_ratio'] = n_pred_s_frame / len(s_seg)
        raw_data[f'v{i+1}_gt_sum_ratio'] = n_gt_s_frame / len(g_seg)
        
    return raw_data