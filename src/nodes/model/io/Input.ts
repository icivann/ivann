import { Node } from '@baklavajs/core';
import { Nodes } from '@/nodes/model/Types';

export default class Input extends Node {
  type = Nodes.Input;
  name = '';

  constructor() {
    super();
    this.addOutputInterface('Output');
  }

  public setName(name: string) {
    this.name = name;
  }
}
