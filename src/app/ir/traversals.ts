import { IInterfaceState, INodeState, IState } from '@baklavajs/core/dist/baklavajs-core/types';
import { GraphNode, ModelNode } from '@/app/ir/mainNodes';
import { Conv1D, Conv2D, ConvConstructor } from '@/app/ir/conv';
import { UUID, Option } from '@/app/util';

class Graph {
  traverseUiToIr(state: IState): Set<GraphNode> {
    const set: Set<GraphNode> = new Set();
    const map: { [id: string]: INodeState } = {};
    for (const node of state.nodes) {
      map[node.id] = node;
    }

    for (const connection of state.connections) {
      const iNode = map[connection.id];
      const constrMap: { [id: string]: any } = this.traverseOptions(iNode.options);
      const inputoutputMap: { [id: string]: any } = this.traverseOptions(iNode.interfaces);
      const mlNode = this.mappingsForNodes(iNode.type, constrMap, iNode.interfaces);
      const gNode = new GraphNode(mlNode, new UUID(iNode.id));
      set.add(gNode);
    }
    return set;
  }

  traverseOptions(options: Array<[string, any]>): { [id: string]: any } {
    const constrMap: { [id: string]: any } = {};
    for (const option of options) {
      constrMap[option[0]] = option[1];
    }
    return constrMap;
  }

  mappingsForNodes(type: string, options: { [id: string]: any }, inout: { [id: string]: any }): ModelNode {
    return new Conv2D(
      options.Filters,
      [options['Kernel Size x'], options['Kernel Size y']],
      [options['Stride x'], options['Stride y']],
      options.Activation,
      options.Padding,
      [options['Weight Initializer'], options['Weight Regularizer']],
      [options['Bias Initializer'], options['Bias Regularizer']],
      inout.Input,
      inout.Output,
    );
  }
}
