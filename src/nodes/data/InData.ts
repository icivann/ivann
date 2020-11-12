import { Node } from '@baklavajs/core';
import { DataNodes, DataCategories } from '@/nodes/data/Types';

export default class InData extends Node {
  type = DataNodes.InData;
  name = DataNodes.InData;

  constructor() {
    super();

    this.addOutputInterface('Output');
  }
}
