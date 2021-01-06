import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum TanhOptions {

}
export default class Tanh extends Node {
  type = ModelNodes.Tanh;
  name = ModelNodes.Tanh;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
