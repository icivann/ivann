import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

import Conv1d from '@/nodes/model/Conv1d';
import Conv2d from '@/nodes/model/Conv2d';
import Conv3d from '@/nodes/model/Conv3d';
import Custom from '@/nodes/model/custom/Custom';
import OutModel from '@/nodes/model/OutModel';
import Concat from '@/nodes/model/operations/Concat';
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
import Linear from '@/nodes/model/Linear';
import Transformer from '@/nodes/model/Transformer';
import Bilinear from '@/nodes/model/Bilinear';
import Softmax from '@/nodes/model/Softmax';
import Softmin from '@/nodes/model/Softmin';

export default class ModelCanvas extends AbstractCanvas {
  public registerNodes(editor: Editor): void {
    editor.registerNodeType(Nodes.Conv1d, Conv1d, Layers.Conv);
    editor.registerNodeType(Nodes.Conv2d, Conv2d, Layers.Conv);
    editor.registerNodeType(Nodes.Conv3d, Conv3d, Layers.Conv);
    editor.registerNodeType(Nodes.ConvTranspose1d, Convtranspose1d, Layers.Conv);
    editor.registerNodeType(Nodes.ConvTranspose2d, Convtranspose2d, Layers.Conv);
    editor.registerNodeType(Nodes.ConvTranspose3d, Convtranspose3d, Layers.Conv);

    editor.registerNodeType(Nodes.MaxPool1d, Maxpool1d, Layers.Pool);
    editor.registerNodeType(Nodes.MaxPool2d, Maxpool2d, Layers.Pool);
    editor.registerNodeType(Nodes.MaxPool3d, Maxpool3d, Layers.Pool);

    editor.registerNodeType(Nodes.Dropout, Dropout, Layers.Dropout);
    editor.registerNodeType(Nodes.Dropout2d, Dropout2d, Layers.Dropout);
    editor.registerNodeType(Nodes.Dropout3d, Dropout3d, Layers.Dropout);

    editor.registerNodeType(Nodes.Relu, ReLU, Layers.Activation);

    editor.registerNodeType(Nodes.Custom, Custom, Layers.Custom);
    editor.registerNodeType(Nodes.InModel, InModel, Layers.IO);
    editor.registerNodeType(Nodes.OutModel, OutModel, Layers.IO);
    editor.registerNodeType(Nodes.Concat, Concat, Layers.Operations);

    editor.registerNodeType(Nodes.Linear, Linear, Layers.Linear);
    editor.registerNodeType(Nodes.Bilinear, Bilinear, Layers.Linear);

    editor.registerNodeType(Nodes.Transformer, Transformer, Layers.Transformer);
    editor.registerNodeType(Nodes.Softmin, Softmin, Layers.NonLinearActivation);
    editor.registerNodeType(Nodes.Softmax, Softmax, Layers.NonLinearActivation);
  }
}
