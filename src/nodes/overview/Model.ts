import { Node } from '@baklavajs/core';
import { Overview } from '@/nodes/model/Types';
import { EditorModel } from '@/store/editors/types';
import { NodeIOChange } from '@/nodes/overview/EditorIOUtils';
import { getEditorIOs } from '@/store/editors/utils';

export default class Model extends Node {
  name = '';
  type = Overview.ModelNode;

  constructor(model?: EditorModel) {
    super();
    if (model) {
      this.name = model.name;
      const { inputs, outputs } = getEditorIOs(model);
      this.updateIO({ added: inputs, removed: [] }, { added: outputs, removed: [] });
    }
  }

  public getCurrentIO(): ({
    inputs: string[];
    outputs: string[];
  }) {
    const inputs: string[] = [];
    const outputs: string[] = [];
    this.interfaces.forEach(((value, key) => {
      (value.isInput ? inputs : outputs).push(key);
    }));
    return { inputs, outputs };
  }

  public updateIO(inputChange: NodeIOChange, outputChange: NodeIOChange) {
    for (const inputAdded of inputChange.added) this.addInputInterface(inputAdded);
    for (const inputRemoved of inputChange.removed) this.removeInterface(inputRemoved);
    for (const outputAdded of outputChange.added) this.addOutputInterface(outputAdded);
    for (const outputRemoved of outputChange.removed) this.removeInterface(outputRemoved);
  }
}
