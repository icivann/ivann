import { Node } from '@baklavajs/core';
import { DataNodes } from '@/nodes/data/Types';

export default class LoadImages extends Node {
  type = DataNodes.LoadImages;
  name = DataNodes.LoadImages;

  constructor() {
    super();

    this.addOutputInterface('Output');
  }
}
