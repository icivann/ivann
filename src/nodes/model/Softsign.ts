import { Node } from '@baklavajs/core';
import { ModelNodes } from '@/nodes/model/Types';
import { TypeOptions } from '@/nodes/model/BaklavaDisplayTypeOptions';
import CheckboxValue from '@/baklava/CheckboxValue';

export enum SoftsignOptions {

}
export default class Softsign extends Node {
  type = ModelNodes.Softsign;
  name = ModelNodes.Softsign;

  constructor() {
    super();
    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
