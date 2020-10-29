import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import Model from '@/nodes/overview/Model';
import { Overview } from '@/nodes/model/Types';

export default class OverviewCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    editor.registerNodeType(Overview.ModelNode, Model as any);
  }
}
