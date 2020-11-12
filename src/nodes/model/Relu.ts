import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';

export enum ReLUOptions {
  Inplace = 'Inplace'
}
export default class ReLU extends Node {
  type = ModelNodes.Relu;
  name = ModelNodes.Relu;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ReLUOptions.Inplace, TypeOptions.TickBoxOption);
  }
}
