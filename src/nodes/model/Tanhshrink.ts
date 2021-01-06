import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum TanhshrinkOptions {

}
export default class Tanhshrink extends Node {
  type = ModelNodes.Tanhshrink;
  name = ModelNodes.Tanhshrink;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
