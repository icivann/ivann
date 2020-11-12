import { Node } from '@baklavajs/core';
import { DataNodes } from '@/nodes/data/Types';

export default class Grayscale extends Node {
  type = DataNodes.Grayscale;
  name = DataNodes.Grayscale;

  constructor() {
    super();

    this.addInputInterface('Input');
    this.addOutputInterface('Output');
  }
}
