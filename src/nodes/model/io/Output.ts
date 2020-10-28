import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';
import { EditorIO, EditorModel } from '@/store/editors/types';

export default class Output extends Node {
  type = Nodes.Output;
  name = '';
  private outputReference: EditorIO[] = [];

  constructor(currEditorModel: EditorModel) {
    super();
    this.addInputInterface('Input');
    if (currEditorModel.outputs) {
      this.outputReference = currEditorModel.outputs;
      const count = this.outputReference.length;
      const outputName = `Output${count}`;
      this.outputReference.push({ name: outputName });
      this.name = outputName;
    }
  }

  public onRemove() {
    for (let i = 0; i < this.outputReference.length; i += 1) {
      if (this.outputReference[i].name === this.name) {
        this.outputReference.splice(i, 1);
      }
    }
  }
}
