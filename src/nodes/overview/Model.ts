import { Node } from '@baklavajs/core';
import { EditorModel } from '@/store/editors/types';
import { NodeIOChange } from '@/nodes/overview/EditorIOUtils';
import { getEditorIOs } from '@/store/editors/utils';
import { INodeState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';
import { OverviewNodes } from '@/nodes/overview/Types';

export default class Model extends Node {
  name = '';
  type = OverviewNodes.ModelNode;

  constructor(model?: EditorModel, overviewFlag?: boolean) {
    super();
    if (model) {
      this.name = model.name;
      // eslint-disable-next-line
      let { inputs, outputs } = getEditorIOs(model);
      // TODO: Clean up hack
      if (overviewFlag) inputs = [];
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

  public load(state: INodeState) {
    const inputs: string[] = [];
    const outputs: string[] = [];
    state.interfaces.forEach(([k, v]) => {
      if ('isInput' in v) (v.isInput ? inputs : outputs).push(k);
    });
    this.updateIO({ added: inputs, removed: [] }, { added: outputs, removed: [] });
    super.load(state);
  }

  public save(): INodeState {
    /* Copied from default Node implementation, but interfaces extended with isInput. */
    const state: INodeState = {
      type: this.type,
      id: this.id,
      name: this.name,
      options: Array.from(this.options.entries())
        .map(([k, o]) => [k, o.value]),
      state: this.state,
      interfaces: Array.from(this.interfaces.entries())
        .map(([k, i]) => [k, { isInput: i.isInput, ...i.save() }]),
    };
    return this.hooks.save.execute(state);
  }
}
