import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

import Conv1d from '@/nodes/model/Conv1dBaklava';
import Conv2d from '@/nodes/model/Conv2dBaklava';
import Conv3d from '@/nodes/model/Conv3dBaklava';
import Custom from '@/nodes/model/custom/Custom';
import OutModel from '@/nodes/model/OutModel';
import Concat from '@/nodes/model/operations/Concat';
import InModel from '@/nodes/model/InModel';
import Convtranspose1d from '@/nodes/model/Convtranspose1dBaklava';
import Convtranspose2d from '@/nodes/model/Convtranspose2dBaklava';
import Convtranspose3d from '@/nodes/model/Convtranspose3dBaklava';

export default class ModelCanvas extends AbstractCanvas {
  public registerNodes(editor: Editor): void {
    editor.registerNodeType(Nodes.Conv1D, Conv1d, Layers.Conv);
    editor.registerNodeType(Nodes.Conv2D, Conv2d, Layers.Conv);
    editor.registerNodeType(Nodes.Conv3D, Conv3d, Layers.Conv);

    editor.registerNodeType(Nodes.Convtranspose1d, Convtranspose1d, Layers.Conv);
    editor.registerNodeType(Nodes.Convtranspose2d, Convtranspose2d, Layers.Conv);
    editor.registerNodeType(Nodes.Convtranspose3d, Convtranspose3d, Layers.Conv);

    editor.registerNodeType(Nodes.Custom, Custom, Layers.Custom);
    editor.registerNodeType(Nodes.InModel, InModel, Layers.IO);
    editor.registerNodeType(Nodes.OutModel, OutModel, Layers.IO);
    editor.registerNodeType(Nodes.Concat, Concat, Layers.Operations);
  }
}
