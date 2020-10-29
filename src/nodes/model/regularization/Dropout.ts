import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';

export enum DropoutOptions {
  Probability = 'Probability',
}

export default class Dropout extends Node {
  type = Nodes.Dropout;
  name = Nodes.Dropout;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption(DropoutOptions.Probability, 'SliderOption', 0.5, undefined, {
      min: 0,
      max: 1,
    });
  }
}
