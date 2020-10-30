import { Node } from '@baklavajs/core';
import { Overview } from '@/nodes/model/Types';
import { EditorModel } from '@/store/editors/types';

export default class Model extends Node {
  name = '';
  type = Overview.ModelNode;

  constructor(model?: EditorModel) {
    super();
    // if (model) {
    //   this.createFromSidebar(model);
    // }
  }

  // private createFromSidebar(model: EditorModel): void {
  //   const { inputs, outputs, name } = model;
  //   this.name = name;
  //
  //   if (inputs) {
  //     for (const input of inputs) {
  //       this.addInputInterface(input.name);
  //     }
  //   }
  //
  //   if (outputs) {
  //     for (const output of outputs) {
  //       this.addOutputInterface(output.name);
  //     }
  //   }
  // }
}
