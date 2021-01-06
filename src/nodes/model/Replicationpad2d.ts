import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ReplicationPad2dOptions {
  Padding = 'Padding'
}
export default class ReplicationPad2d extends Node {
  type = ModelNodes.ReplicationPad2d;
  name = ModelNodes.ReplicationPad2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ReplicationPad2dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0, 0]);
  }
}
