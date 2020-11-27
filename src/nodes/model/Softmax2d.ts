import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum Softmax2dOptions {

}
export default class Softmax2d extends Node {
  type = ModelNodes.Softmax2d;
  name = ModelNodes.Softmax2d;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
