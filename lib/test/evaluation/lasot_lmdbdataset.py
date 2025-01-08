from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.utils.lmdb_utils import *

'''2021.1.27 LaSOT dataset using lmdb data'''


class LaSOTlmdbDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasot_lmdb_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = str('{}/{}/groundtruth.txt'.format(class_name, sequence_name))
        # decode the groundtruth
        gt_str_list = decode_str(self.base_path, anno_path).split('\n')[:-1]  # the last line is empty
        gt_list = [list(map(float, line.split(','))) for line in gt_str_list]
        ground_truth_rect = np.array(gt_list).astype(np.float64)
        # decode occlusion file
        occlusion_label_path = str('{}/{}/full_occlusion.txt'.format(class_name, sequence_name))
        occ_list = list(map(int, decode_str(self.base_path, occlusion_label_path).split(',')))
        full_occlusion = np.array(occ_list).astype(np.float64)
        # decode out of view file
        out_of_view_label_path = str('{}/{}/out_of_view.txt'.format(class_name, sequence_name))
        out_of_view_list = list(map(int, decode_str(self.base_path, out_of_view_label_path).split(',')))
        out_of_view = np.array(out_of_view_list).astype(np.float64)

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/img'.format(class_name, sequence_name)

        frames_list = [[self.base_path, '{}/{:08d}.jpg'.format(frames_path, frame_number)] for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = ['mouse-1',
                         'mouse-8',
                         'mouse-9',
                         'mouse-17',
                         'electricfan-1',
                         'electricfan-10',
                         'electricfan-18',
                         'electricfan-20',
                         'gecko-1',
                         'gecko-5',
                         'gecko-16',
                         'gecko-19',
                         'dualsense-1',
                        ]
        return sequence_list
