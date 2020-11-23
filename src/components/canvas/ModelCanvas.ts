import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { ModelCategories, ModelNodes } from '@/nodes/model/Types';

import Conv1d from '@/nodes/model/Conv1d';
import Conv2d from '@/nodes/model/Conv2d';
import Conv3d from '@/nodes/model/Conv3d';
import OutModel from '@/nodes/model/OutModel';
import Concat from '@/nodes/model/Concat';
import InModel from '@/nodes/model/InModel';
import Convtranspose1d from '@/nodes/model/Convtranspose1d';
import Convtranspose2d from '@/nodes/model/Convtranspose2d';
import Convtranspose3d from '@/nodes/model/Convtranspose3d';
import Maxpool1d from '@/nodes/model/Maxpool1d';
import Maxpool2d from '@/nodes/model/Maxpool2d';
import Maxpool3d from '@/nodes/model/Maxpool3d';
import Dropout from '@/nodes/model/Dropout';
import Dropout2d from '@/nodes/model/Dropout2d';
import Dropout3d from '@/nodes/model/Dropout3d';
import ReLU from '@/nodes/model/Relu';
import Transformer from '@/nodes/model/Transformer';
import Softmin from '@/nodes/model/Softmin';
import Softmax from '@/nodes/model/Softmax';
import Bilinear from '@/nodes/model/Bilinear';
import Linear from '@/nodes/model/Linear';
import { Editor } from '@baklavajs/core';
import { CommonNodes } from '@/nodes/common/Types';
import Custom from '@/nodes/common/Custom';

export default class ModelCanvas extends AbstractCanvas {
  public nodeList = [
    {
      category: ModelCategories.Conv,
      nodes: [
        {
          name: ModelNodes.Conv1d,
          node: Conv1d,
        },
        {
          name: ModelNodes.Conv2d,
          node: Conv2d,
        },
        {
          name: ModelNodes.Conv3d,
          node: Conv3d,
        },
        {
          name: ModelNodes.ConvTranspose1d,
          node: Convtranspose1d,
        },
        {
          name: ModelNodes.ConvTranspose2d,
          node: Convtranspose2d,
        },
        {
          name: ModelNodes.ConvTranspose3d,
          node: Convtranspose3d,
        },
      ],
    },
    {
      category: ModelCategories.Pool,
      nodes: [
        {
          name: ModelNodes.MaxPool1d,
          node: Maxpool1d,
        },
        {
          name: ModelNodes.MaxPool2d,
          node: Maxpool2d,
        },
        {
          name: ModelNodes.MaxPool3d,
          node: Maxpool3d,
        },
      ],
    },
    {
      category: ModelCategories.Dropout,
      nodes: [
        {
          name: ModelNodes.Dropout,
          node: Dropout,
        },
        {
          name: ModelNodes.Dropout2d,
          node: Dropout2d,
        },
        {
          name: ModelNodes.Dropout3d,
          node: Dropout3d,
        },
      ],
    },
    {
      category: ModelCategories.Activation,
      nodes: [
        {
          name: ModelNodes.Relu,
          node: ReLU,
        },
      ],
    },
    {
      category: ModelCategories.Operations,
      nodes: [
        {
          name: ModelNodes.Concat,
          node: Concat,
        },
      ],
    },
    {
      category: ModelCategories.Linear,
      nodes: [
        {
          name: ModelNodes.Linear,
          node: Linear,
        },
        {
          name: ModelNodes.Bilinear,
          node: Bilinear,
        },
      ],
    },
    {
      category: ModelCategories.Transformer,
      nodes: [
        {
          name: ModelNodes.Transformer,
          node: Transformer,
        },
      ],
    },
    {
      category: ModelCategories.NonLinearActivation,
      nodes: [
        {
          name: ModelNodes.Softmin,
          node: Softmin,
        },
        {
          name: ModelNodes.Softmax,
          node: Softmax,
        },
      ],
    },
  ];

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(ModelNodes.InModel, InModel);
    editor.registerNodeType(ModelNodes.OutModel, OutModel);
  }
}
