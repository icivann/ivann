import { Node } from '@baklavajs/core';
import { DataNodes } from '@/nodes/data/Types';

class OutData extends Node {
  type = DataNodes.OutData;
  name = DataNodes.OutData;

  constructor() {
    super();

    this.addInputInterface('Input');
    this.addInputInterface('Label');
  }
}

export default OutData;
