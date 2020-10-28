import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/train/Types';
import { valuesOf } from '@/app/util';

export enum LossOptions {
  Loss = 'Loss',
}
export enum LossFunctions {
  NLL='nll_loss',
  MSE='mse_loss',
}

export default class DatasetInput extends Node {
  type = Nodes.DatasetInput;
  name = Nodes.DatasetInput;

  constructor() {
    super();
    this.addOption(LossOptions.Loss, 'DropdownOption', 'Valid', undefined, {
      items: valuesOf(LossFunctions),
    });
    this.addInputInterface('pred');
    this.addInputInterface('target');
    this.addOutputInterface('loss');
  }
}
