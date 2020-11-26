import { OverviewNodes } from '@/nodes/overview/Types';
import { EditorModel } from '@/store/editors/types';
import { Node } from '@baklavajs/core';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import { getEditorIOs } from '@/store/editors/utils';
import { INodeState } from '@baklavajs/core/dist/baklavajs-core/types/state.d';

export enum DataOptions {
  BatchSize = 'BatchSize',
}

export class Data extends Node {
  name = '';
  type = OverviewNodes.DataNode;

  constructor(model?: EditorModel) {
    super();

    if (model) {
      this.name = model.name;

      const { inputs } = getEditorIOs(model);

      inputs.forEach((node) => {
        this.addOption(`${node} path`, TypeOptions.TextAreaOption, '');
      });
    }

    this.addOption(DataOptions.BatchSize, TypeOptions.IntOption, 32);
    this.addOutputInterface('Dataset');
  }

  public load(state: INodeState) {
    state.options.forEach(([k, v]) => {
      if (k.endsWith('path')) {
        this.addOption(k, TypeOptions.TextAreaOption, v);
      }
    });
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
