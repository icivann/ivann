import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum ReplicationPad3dOptions {
  Padding = 'Padding'
}
export default class ReplicationPad3d extends Node {
  type = ModelNodes.ReplicationPad3d;
  name = ModelNodes.ReplicationPad3d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
    this.addOption(ReplicationPad3dOptions.Padding, TypeOptions.VectorOption, [0, 0, 0, 0, 0, 0]);
  }
}
