import { Node } from '@baklavajs/core';
import { Nodes, Overview } from '@/nodes/model/Types';

export default class TrainClassifier extends Node {
  type = Overview.TrainNode;
  name = Nodes.TrainClassifier;

  constructor() {
    super();
  }
}
