import glob
import json
import os.path
from pathlib import Path
import sys

import cv2
import numpy as np
import inspect
import importlib
import random

import openslide
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# ---->
from sklearn.metrics import roc_curve
from xml.dom import minidom

from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch

# ---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# ---->
import pytorch_lightning as pl

plt.rcParams.update({
    # 'font.family': 'Times New Roman',  # 新罗马字体
    'font.size': 20,  # 字体大小
    'font.weight': 'bold'  # 字体加粗
})

plt.figure(figsize=(10, 8))


class ModelInterface(pl.LightningModule):

    # ---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        # print(kargs)
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']
        # self.heatmap = heatmap

        # ---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # ---->Metrics
        if self.n_classes > 2:
            self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average='macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes=self.n_classes),
                                                     torchmetrics.F1Score(num_classes=self.n_classes,
                                                                          average='macro'),
                                                     torchmetrics.Recall(average='macro',
                                                                         num_classes=self.n_classes),
                                                     torchmetrics.Precision(average='macro',
                                                                            num_classes=self.n_classes),
                                                     torchmetrics.Specificity(average='macro',
                                                                              num_classes=self.n_classes)])
        else:
            self.AUROC = torchmetrics.AUROC(num_classes=2, average='macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=2,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(num_classes=2),
                                                     torchmetrics.F1Score(num_classes=2,
                                                                          average='macro'),
                                                     torchmetrics.Recall(average='macro',
                                                                         num_classes=2),
                                                     torchmetrics.Precision(average='macro',
                                                                            num_classes=2),
                                                     torchmetrics.Specificity(average='macro',
                                                                              num_classes=2)])
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        # --->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0

    # ---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def training_step(self, batch, batch_idx):
        # ---->inference
        data, label, coords, slide_id = batch
        results_dict = self.model(data=data, label=label, coords=coords, slide_id=slide_id)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # ---->loss
        loss = self.loss(logits, label)

        # ---->acc log
        Y_hat = int(Y_hat)
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def validation_step(self, batch, batch_idx):
        data, label, _, _ = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        # ---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label}

    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim=0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim=0)

        # ---->
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()),
                      on_epoch=True, logger=True)

        # ---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        # ---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count + 1
            random.seed(self.count * 50)

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        # if batch_idx == 117:
        #     print(batch_idx)
        data, label, coords, slide_id = batch
        slide_id1 = slide_id[0]
        # print(slide_id)
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        # feature = results_dict['feature']
        # os.makedirs('/home/ubuntu/zhonghaiqin/code_wsi/TransMIL-main/logs/config/TCGA-Lung/TDTMIL_TCGA-Lung/fold0/feature', exist_ok=True)
        # torch.save(feature, Path(
        #             '/home/ubuntu/zhonghaiqin/code_wsi/TransMIL-main/logs/config/TCGA-Lung/TDTMIL_TCGA-Lung/fold0/feature') / f'{slide_id1}.pt')

        if self.hparams.cfg['General']['heatmap']:
            attention = results_dict['weights'].cpu().numpy().reshape(-1)
            coords = coords.cpu().numpy().squeeze()
            slide_id = slide_id[0]
            if 'camelyon16' in self.hparams.cfg['Data']['data_dir']:
                annotation_dir = Path(
                    '/home/ubuntu/zhonghaiqin/dataset/CAMELYON16/testing/annotation') / f'{slide_id}.xml'
                if annotation_dir.exists():
                    contours = load_annotations_xml(str(annotation_dir))
                else:
                    contours = None
                slide_path = Path('/home/ubuntu/zhonghaiqin/dataset/CAMELYON16/testing/images') / f'{slide_id}.tif'

            elif 'TCGA-Lung' in self.hparams.cfg['Data']['data_dir']:
                contours = None
                slide_path = Path('/home/ubuntu/zhonghaiqin/dataset/TCGA-Lung/*') / f'{slide_id}.svs'
                # slide_path = glob.glob(slide_path)[0]
            else:
                contours = None
                slide_path = None
            if slide_path:
                heatmap = create_heatmap(slide_path, attention, slide_level=-1, contours=contours, coords=coords)
                save_path = Path(self.log_path) / f'heatmap'
                os.makedirs(save_path, exist_ok=True)
                cv2.imwrite(str(Path(save_path) / f'{slide_id}.png'), heatmap)

        # ---->acc log
        Y = int(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label,
                'slide_id': slide_id1}
    def test_epoch_end(self, output_results):
        # label_mapping = {
        #     "0": "Normal",
        #     "1": "Tumor"
        # }
        # label_mapping = {
        #     "0": "LUAD",
        #     "1": "LUSC"
        # }
        probs = torch.cat([x['Y_prob'] for x in output_results], dim=0)
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results], dim=0)
        # features = torch.cat([x['feature'] for x in output_results], dim=0).cpu().numpy()
        slide_id = [x['slide_id'] for x in output_results]
        labels = target.cpu().numpy()
        # tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=2000, random_state=42)
        # features_tsne = tsne.fit_transform(features)
        labels = np.squeeze(np.array(labels))
        pred_dict = {'slide_id': slide_id, 'label': labels, 'pred': np.squeeze(max_probs.cpu().numpy()).tolist(),
                     'p_0': probs.cpu().numpy().T[0].tolist(), 'p_1': probs.cpu().numpy().T[1].tolist()}
        df = pd.DataFrame(pred_dict)
        df.to_csv(os.path.join(self.log_path, 'pred.csv'), index=False)
        # for label in range(len(label_mapping)):
        #     plt.scatter(features_tsne[labels == label, 0], features_tsne[labels == label, 1],
        #                 label=label_mapping[str(label)], alpha=0.5)
        # plt.title('TDTMIL', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        # plt.savefig(os.path.join(self.log_path, 'TDTMIL.png'), dpi=600)
        # plt.close()

        roc_data = {}
        # ---->
        auc = self.AUROC(probs, target.squeeze())
        fpr, tpr, threshold = roc_curve(target.cpu().numpy(), probs[:, -1].cpu().numpy(), pos_label=1)

        print(len(fpr))
        roc_data.update({'fpr': list(fpr)})
        roc_data.update({'tpr': list(tpr)})
        with open(os.path.join(self.log_path, 'roc.json'), 'w') as file:
            json.dump(roc_data, file)
        file.close()
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics['auc'] = auc
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        print()
        # ---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0:
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        # ---->
        result = pd.DataFrame([metrics])
        print(self.log_path)
        result.to_csv(self.log_path / 'result.csv')

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        # print(camel_name)
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)

        except:

            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = list(inspect.signature(Model.__init__).parameters.keys())[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)


def create_heatmap(slide_path, attention, slide_level=-1, contours=None, coords=None):
    """
    create the heatmap of WSI with coord json file and attention scores of patches.

    :param slide_path: WSI filepath
    :param attention: attention scores of patches
    :param slide_level: the sample level of slide
    :param contours: the contours of ROI's annotation
    :return: a heatmap
    """
    # read some necessary variables from json file of coord
    # coord_dict = load_json(coord_filepath)
    # slide_filepath = c
    # num_patches = coord_dict['num_patches']
    # coords = coord_dict['coords']
    # patch_size_level0 = coord_dict['patch_size_level0']

    slide = openslide.open_slide(slide_path)
    if slide_level < 0:
        slide_level = slide.get_best_level_for_downsample(32)
    thumbnail = slide.get_thumbnail(slide.level_dimensions[slide_level]).convert('RGB')
    thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)

    level_downsample = slide.level_downsamples[slide_level]
    # assert num_patches == len(coords) == len(attention), f"{num_patches}-{len(coords)}-{len(attention)}"
    assert len(attention) == len(coords), f"{len(attention)}-{len(coords)}"

    # scale the attention to [0, 1] and create the color mapping
    attention = np.uint8(255 * ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))))
    colors = cv2.applyColorMap(attention, cv2.COLORMAP_JET)

    # create the blank heatmap with white background
    heatmap = np.ones(thumbnail.shape, dtype=np.uint8) * 255
    for i in range(len(attention)):
        top_left_x, top_left_y = int(coords[i][0]), int(coords[i][1])
        points = get_three_points(top_left_x / level_downsample, top_left_y / level_downsample, 256 / level_downsample)
        c = (int(colors[i, 0, 0]), int(colors[i, 0, 1]), int(colors[i, 0, 2]))
        # draw the rectangle filled with attention color
        cv2.rectangle(heatmap, points[0], points[1], color=c, thickness=-1)
    # mix the heatmap and slide thumbnail
    heatmap = cv2.addWeighted(heatmap, 0.5, thumbnail, 0.5, 0)

    # draw the ROI region if contours exist
    if contours is not None:
        contours = [np.asarray(c / level_downsample).astype(np.int32) for c in contours]
        heatmap = cv2.drawContours(heatmap, contours, -1, (0, 255, 255), thickness=1)

    return heatmap


def get_three_points(x_step, y_step, size):
    top_left = (int(x_step), int(y_step))
    bottom_right = (int(top_left[0] + size), int(top_left[1] + size))
    center = (int((top_left[0] + bottom_right[0]) // 2), int((top_left[1] + bottom_right[1]) // 2))
    return top_left, bottom_right, center


def load_annotations_xml(annotations_xml):
    dom = minidom.parse(annotations_xml)
    root = dom.documentElement
    annotations = root.getElementsByTagName('Annotation')

    contours = []
    for a in annotations:
        coords = a.getElementsByTagName('Coordinates')[0].getElementsByTagName('Coordinate')
        contour = np.array([[c.getAttribute('X'), c.getAttribute('Y')] for c in coords], dtype=np.float64)
        contour = np.expand_dims(contour, 1)
        contours.append(contour)
        # print(contour.shape)
    return contours
