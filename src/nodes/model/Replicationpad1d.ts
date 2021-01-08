import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ReplicationPad1dOptions {
  Padding = 'Padding'
}
export default class ReplicationPad1d extends Node {
  type = ModelNodes.ReplicationPad1d;
  name = ModelNodes.ReplicationPad1d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ReplicationPad1dOptions.Padding, TypeOptions.VectorOption, [0, 0]);
  }
}
