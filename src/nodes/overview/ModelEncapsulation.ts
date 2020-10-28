import { Node } from '@baklavajs/core';
import { Overview } from '@/nodes/model/Types';
import { EditorIO } from '@/store/editors/types';
import editorIOPartition from '@/nodes/overview/EditorIOUtils';

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

    if (inputs) {
      const { added, removed } = editorIOPartition(inputs, this.inputs);
      let modified = false;
      if (added.length > 0) {
        for (const input of added) {
          this.addInputInterface(input.name);
        }
        modified = true;
      }
      if (removed.length > 0) {
        for (const input of removed) {
          this.removeInterface(input.name);
        }
        modified = true;
      }
      if (modified) this.inputs = inputs.slice();
    }

    if (outputs) {
      const { added, removed } = editorIOPartition(outputs, this.outputs);
      let modified = false;
      if (added.length > 0) {
        for (const output of added) {
          this.addOutputInterface(output.name);
        }
        modified = true;
      }
      if (removed.length > 0) {
        for (const output of removed) {
          this.removeInterface(output.name);
        }
        modified = true;
      }
      if (modified) this.outputs = outputs.slice();
    }
  }
}
