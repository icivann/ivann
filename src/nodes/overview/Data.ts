import { OverviewNodes } from '@/nodes/overview/Types';
import { EditorModel } from '@/store/editors/types';
import { Node } from '@baklavajs/core';

export class Data extends Node {
  name = '';
  type = OverviewNodes.DataNode;

  constructor(model?: EditorModel) {
    super();
    if (model) {
      this.name = model.name;
    }

    this.addOutputInterface('Output');
    this.addOutputInterface('Labels');
  }
}
