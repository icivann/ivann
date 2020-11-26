import { OverviewNodes } from '@/nodes/overview/Types';
import { EditorModel } from '@/store/editors/types';
import { Node } from '@baklavajs/core';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import { getEditorIOs } from '@/store/editors/utils';

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
}
