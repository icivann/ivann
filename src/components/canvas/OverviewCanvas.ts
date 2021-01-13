import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import Model from '@/nodes/overview/Model';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import TrainClassifier from '@/nodes/overview/train/TrainClassifier';
import Adadelta from '@/nodes/overview/optimizers/Adadelta';

import { Data } from '@/nodes/overview/Data';
import { Editor } from '@baklavajs/core';
import OverviewCustom from '@/nodes/overview/OverviewCustom';
import { CommonNodes } from '@/nodes/common/Types';

import TripletMarginLoss from '@/nodes/overview/loss/Tripletmarginloss';
import MultiMarginLoss from '@/nodes/overview/loss/Multimarginloss';
import CosineEmbeddingLoss from '@/nodes/overview/loss/Cosineembeddingloss';
import MultiLabelSoftMarginLoss from '@/nodes/overview/loss/Multilabelsoftmarginloss';
import SmoothL1Loss from '@/nodes/overview/loss/Smoothl1loss';
import MultiLabelMarginLoss from '@/nodes/overview/loss/Multilabelmarginloss';
import HingeEmbeddingLoss from '@/nodes/overview/loss/Hingeembeddingloss';
import MarginRankingLoss from '@/nodes/overview/loss/Marginrankingloss';
import BCEWithLogitsLoss from '@/nodes/overview/loss/Bcewithlogitsloss';
import BCELoss from '@/nodes/overview/loss/Bceloss';
import KLDivLoss from '@/nodes/overview/loss/Kldivloss';
import PoissonNLLLoss from '@/nodes/overview/loss/Poissonnllloss';
import NLLLoss from '@/nodes/overview/loss/Nllloss';
import CTCLoss from '@/nodes/overview/loss/Ctcloss';
import CrossEntropyLoss from '@/nodes/overview/loss/Crossentropyloss';
import L1Loss from '@/nodes/overview/loss/L1loss';
import MSELoss from '@/nodes/overview/loss/Mseloss';
import Adamax from '@/nodes/overview/optimizers/Adamax';
import SparseAdam from '@/nodes/overview/optimizers/Sparseadam';
import AdamW from '@/nodes/overview/optimizers/Adamw';
import Adagrad from '@/nodes/overview/optimizers/Adagrad';
import ASGD from '@/nodes/overview/optimizers/Asgd';
import LBFGS from '@/nodes/overview/optimizers/Lbfgs';
import RMSprop from '@/nodes/overview/optimizers/Rmsprop';
import Rprop from '@/nodes/overview/optimizers/Rprop';
import SGD from '@/nodes/overview/optimizers/Sgd';
import Adam from '@/nodes/overview/optimizers/Adam';
import TrainGAN from '@/nodes/overview/train/TrainGAN';

export default class OverviewCanvas extends AbstractCanvas {
  nodeList = [
    {
      category: OverviewCategories.Train,
      nodes: [
        {
          name: OverviewNodes.TrainClassifier,
          node: TrainClassifier,
          img: 'train-icon.svg',
        },

        {
          name: OverviewNodes.TrainGAN,
          node: TrainGAN,
        },
      ],
    },
    {
      category: OverviewCategories.LossFunctions,
      nodes: [

        {
          name: OverviewNodes.L1Loss,
          node: L1Loss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.MSELoss,
          node: MSELoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.CrossEntropyLoss,
          node: CrossEntropyLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.CTCLoss,
          node: CTCLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.NLLLoss,
          node: NLLLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.PoissonNLLLoss,
          node: PoissonNLLLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.KLDivLoss,
          node: KLDivLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.BCELoss,
          node: BCELoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.BCEWithLogitsLoss,
          node: BCEWithLogitsLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.MarginRankingLoss,
          node: MarginRankingLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.HingeEmbeddingLoss,
          node: HingeEmbeddingLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.MultiLabelMarginLoss,
          node: MultiLabelMarginLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.SmoothL1Loss,
          node: SmoothL1Loss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.MultiLabelSoftMarginLoss,
          node: MultiLabelSoftMarginLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.CosineEmbeddingLoss,
          node: CosineEmbeddingLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.MultiMarginLoss,
          node: MultiMarginLoss,
          img: 'loss-icon.svg',
        },

        {
          name: OverviewNodes.TripletMarginLoss,
          node: TripletMarginLoss,
          img: 'loss-icon.svg',
        },
      ],
    },
    {
      category: OverviewCategories.Optimizer,
      nodes: [
        {
          name: OverviewNodes.Adadelta,
          node: Adadelta,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.Adamax,
          node: Adamax,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.SparseAdam,
          node: SparseAdam,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.AdamW,
          node: AdamW,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.Adam,
          node: Adam,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.Adagrad,
          node: Adagrad,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.ASGD,
          node: ASGD,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.LBFGS,
          node: LBFGS,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.RMSprop,
          node: RMSprop,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.Rprop,
          node: Rprop,
          img: 'optimiser-icon.svg',
        },

        {
          name: OverviewNodes.SGD,
          node: SGD,
          img: 'optimiser-icon.svg',
        },

      ],
    },
  ];

  customNodeType = OverviewCustom;
  customNodeName = OverviewNodes.Custom;

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(OverviewNodes.ModelNode, Model);
    editor.registerNodeType(OverviewNodes.DataNode, Data);
  }
}
