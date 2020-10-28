import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';
import Dense from '@/nodes/model/linear/Dense';
import MaxPool2D from '@/nodes/model/pool/MaxPool2D';
import Dropout from '@/nodes/model/regularization/Dropout';
import Flatten from '@/nodes/model/reshape/Flatten';
import Custom from '@/nodes/model/custom/Custom';
import Conv1D from '@/nodes/model/conv/Conv1D';
import Conv2D from '@/nodes/model/conv/Conv2D';
import Conv3D from '@/nodes/model/conv/Conv3D';
import Output from '@/nodes/model/io/Output';
import Input from '@/nodes/model/io/Input';

export default class ModelCanvas extends AbstractCanvas {
  public registerNodes(editor: Editor): void {
    editor.registerNodeType(Nodes.Dense, Dense, Layers.Linear);
    editor.registerNodeType(Nodes.Conv1D, Conv1D, Layers.Conv);
    editor.registerNodeType(Nodes.Conv2D, Conv2D, Layers.Conv);
    editor.registerNodeType(Nodes.Conv3D, Conv3D, Layers.Conv);
    editor.registerNodeType(Nodes.MaxPool2D, MaxPool2D, Layers.Pool);
    editor.registerNodeType(Nodes.Dropout, Dropout, Layers.Regularization);
    editor.registerNodeType(Nodes.Flatten, Flatten, Layers.Reshape);
    editor.registerNodeType(Nodes.Custom, Custom, Layers.Custom);
    editor.registerNodeType(Nodes.Output, Output as any, Layers.Core);
    editor.registerNodeType(Nodes.Input, Input as any, Layers.Core);

    editor.events.removeNode.addListener(this, (node) => this.onRemove(node));
  }

  private onRemove = (node: any) => {
    if (node.type === Nodes.Input) {
      (node as Input).onRemove();
    } else if (node.type === Nodes.Output) {
      (node as Output).onRemove();
    }
  }
}
