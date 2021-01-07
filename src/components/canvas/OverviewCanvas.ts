import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import Model from '@/nodes/overview/Model';
import { OverviewCategories, OverviewNodes } from '@/nodes/overview/Types';
import TrainClassifier from '@/nodes/overview/train/TrainClassifier';
import Adadelta from '@/nodes/overview/optimizers/Adadelta';

import { Data } from '@/nodes/overview/Data';
import { Editor } from '@baklavajs/core';
import OverviewCustom from '@/nodes/overview/OverviewCustom';

import Embedding from '@/nodes/model/Embedding';
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

export default class OverviewCanvas extends AbstractCanvas {
  nodeList = [
    {
      category: OverviewCategories.Train,
      nodes: [
        {
          name: OverviewNodes.TrainClassifier,
          node: TrainClassifier,
        },
      ],
    },
    {
      category: OverviewCategories.LossFunctions,
      nodes: [

        {
          name: OverviewNodes.L1Loss,
          node: L1Loss,
        },

        {
          name: OverviewNodes.MSELoss,
          node: MSELoss,
        },

        {
          name: OverviewNodes.CrossEntropyLoss,
          node: CrossEntropyLoss,
        },

        {
          name: OverviewNodes.CTCLoss,
          node: CTCLoss,
        },

        {
          name: OverviewNodes.NLLLoss,
          node: NLLLoss,
        },

        {
          name: OverviewNodes.PoissonNLLLoss,
          node: PoissonNLLLoss,
        },

        {
          name: OverviewNodes.KLDivLoss,
          node: KLDivLoss,
        },

        {
          name: OverviewNodes.BCELoss,
          node: BCELoss,
        },

        {
          name: OverviewNodes.BCEWithLogitsLoss,
          node: BCEWithLogitsLoss,
        },

        {
          name: OverviewNodes.MarginRankingLoss,
          node: MarginRankingLoss,
        },

        {
          name: OverviewNodes.HingeEmbeddingLoss,
          node: HingeEmbeddingLoss,
        },

        {
          name: OverviewNodes.MultiLabelMarginLoss,
          node: MultiLabelMarginLoss,
        },

        {
          name: OverviewNodes.SmoothL1Loss,
          node: SmoothL1Loss,
        },

        {
          name: OverviewNodes.MultiLabelSoftMarginLoss,
          node: MultiLabelSoftMarginLoss,
        },

        {
          name: OverviewNodes.CosineEmbeddingLoss,
          node: CosineEmbeddingLoss,
        },

        {
          name: OverviewNodes.MultiMarginLoss,
          node: MultiMarginLoss,
        },

        {
          name: OverviewNodes.TripletMarginLoss,
          node: TripletMarginLoss,
        },
      ],
    },
    {
      category: OverviewCategories.Optimizer,
      nodes: [
        {
          name: OverviewNodes.Adadelta,
          node: Adadelta,
        },
      ],
    },
  ];

  customNodeType = OverviewCustom;

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(OverviewNodes.ModelNode, Model);
    editor.registerNodeType(OverviewNodes.DataNode, Data);
  }
}
