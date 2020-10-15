import { Node } from '@baklavajs/core';
import { Layers, Nodes } from '@/nodes/model/Types';

export default class Dropout extends Node {
  type = Layers.Regularization
  name = Nodes.Dropout;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');

    this.addOption('Probability', 'SliderOption', 0.5, undefined, {
      min: 0,
      max: 1,
    });
  }
}
