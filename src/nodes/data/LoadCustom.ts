import { Node } from '@baklavajs/core';
import { DataNodes } from '@/nodes/data/Types';

export default class LoadCustom extends Node {
  type = DataNodes.LoadCustom;
  name = DataNodes.LoadCustom;

  constructor() {
    super();

    this.addOutputInterface('Output');
  }
}
