import { Node } from '@baklavajs/core';
import { Overview } from '@/nodes/model/Types';
import { EditorIO } from '@/store/editors/types';

export interface ModelOption {
  name: string;
  inputs?: EditorIO[];
  outputs?: EditorIO[];
}

export default class Model extends Node {
  name = '';
  type = Overview.ModelNode;

  constructor(model?: ModelOption) {
    super();
    if (model) {
      this.createFromSidebar(model);
    }
  }

  private createFromSidebar(model: ModelOption): void {
    const { inputs, outputs, name } = model;
    this.name = name;

    if (inputs) {
      for (const input of inputs) {
        this.addInputInterface(input.name);
      }
    }

    if (outputs) {
      for (const output of outputs) {
        this.addOutputInterface(output.name);
      }
    }
  }
}
