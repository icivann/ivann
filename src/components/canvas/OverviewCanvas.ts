import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { Editor } from '@baklavajs/core';
import ModelEncapsulation from '@/nodes/overview/ModelEncapsulation';
import { Overview } from '@/nodes/model/Types';

export default class OverviewCanvas extends AbstractCanvas {
  registerNodes(editor: Editor): void {
    editor.registerNodeType(Overview.ModelNode, ModelEncapsulation as any);
  }
}
