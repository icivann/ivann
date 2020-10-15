import { IInterfaceState, INodeState, IState } from '@baklavajs/core/dist/baklavajs-core/types';
import { ModelNode } from '@/app/ir/mainNodes';
import { UUID, Option } from '@/app/util';
import GraphNode from '@/app/ir/GraphNode';
import Conv2D from '@/app/ir/Conv2D';

class Graph {
  traverseUiToIr(state: IState): Set<GraphNode> {
    const set: Set<GraphNode> = new Set();
    const inputMap: { [id: string]: UUID } = {};
    const outputMap: { [id: string]: UUID } = {};
    for (const node of state.nodes) {
      const inputoutputMap: { [id: string]: any } = this.traverseOptions(node.interfaces);
      inputMap[inputoutputMap.Input] = new UUID(node.id);
      outputMap[inputoutputMap.Output] = new UUID(node.id);
    }

    for (const node of state.nodes) {
      const constrMap: { [id: string]: any } = this.traverseOptions(node.options);
      const inputOutputMap: { [id: string]: any } = this.traverseOutputs(node.interfaces);
      const mlNode = this.mappingsForNodes(node.name, constrMap, inputMap, outputMap, inputOutputMap);
      const gNode = new GraphNode(mlNode, new UUID(node.id));
      set.add(gNode);
    }
    return set;
  }

  traverseOptions(options: Array<[string, any]>): { [id: string]: any } {
    const constrMap: { [id: string]: any } = {};
    for (const option of options) {
      constrMap[option[0]] = option[1].id;
    }
    return constrMap;
  }

  traverseOutputs(interfaces: Array<[string, any]>): { [id: string]: Array<string> } {
    const inoutMap: { [id: string]: Array<string> } = {};
    for (const interf of interfaces) {
      if (inoutMap[interf[0]] === undefined) {
        inoutMap[interf[0]] = [interf[1].id];
      } else {
        inoutMap[interf[0]].push(interf[1].id);
      }
    }
    return inoutMap;
  }

  mappingsForNodes(type: string, options: { [id: string]: any },
    inputMap: { [id: string]: UUID }, outputMap: { [id: string]: UUID },
    inputOutputMap: { [id: string]: Array<string> }): ModelNode {
    const outputs = new Set<UUID>();
    for (const output of inputOutputMap.Output) {
      outputs.add(outputMap[output]);
    }
    return new Conv2D(
      outputs,
      options.Filters,
      options.Padding,
      [options['Kernel Size x'], options['Kernel Size y']],
      [options['Stride x'], options['Stride y']],
      inputMap[inputOutputMap.Input[0]],
      options.Activation,
      [options['Weight Initializer'], options['Weight Regularizer']],
      [options['Bias Initializer'], options['Bias Regularizer']],
    );
  }
}
