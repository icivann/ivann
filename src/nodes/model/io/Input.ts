import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';
import { EditorIO, EditorModel } from '@/store/editors/types';

export default class Input extends Node {
  type = Nodes.Input;
  name = '';
  private inputReference: EditorIO[] = [];

  public constructor(currEditorModel: EditorModel) {
    super();
    this.addOutputInterface('Output');
    if (currEditorModel.inputs) {
      this.inputReference = currEditorModel.inputs;
      const count = this.inputReference.length;
      const inputName = `Input${count}`;
      this.inputReference.push({ name: inputName });
      this.name = inputName;
    }
  }

  public onRemove() {
    for (let i = 0; i < this.inputReference.length; i += 1) {
      if (this.inputReference[i].name === this.name) {
        this.inputReference.splice(i, 1);
      }
    }
  }
}
