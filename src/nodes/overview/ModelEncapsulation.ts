import { Node } from '@baklavajs/core';
import { Overview } from '@/nodes/model/Types';
import { EditorIO } from '@/store/editors/types';
import editorIOEquals from '@/nodes/overview/EditorIOUtils';

export interface ModelOption {
  name: string;
  inputs?: EditorIO[];
  outputs?: EditorIO[];
}

export default class ModelEncapsulation extends Node {
  name = '';
  type = Overview.ModelNode;

  private inputs: EditorIO[] = [];
  private outputs: EditorIO[] = [];

  constructor(model?: ModelOption) {
    super();
    if (model) {
      this.update(model);
    }
  }

  public update(model: ModelOption): void {
    const { inputs, outputs, name } = model;
    this.name = name;

    if (inputs && !editorIOEquals(inputs, this.inputs)) {
      for (const input of this.inputs) {
        this.removeInterface(input.name);
      }
      this.inputs = inputs.slice();
      for (const input of this.inputs) {
        this.addInputInterface(input.name);
      }
    }

    if (outputs && !editorIOEquals(outputs, this.outputs)) {
      for (const output of this.outputs) {
        this.removeInterface(output.name);
      }
      this.outputs = outputs.slice();
      for (const output of this.outputs) {
        this.addOutputInterface(output.name);
      }
    }
  }
}
