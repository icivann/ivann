import {GraphNode} from "@/app/ir/mainNodes";
import {Conv2D} from "@/app/ir/conv";
import {UUID} from "@/app/util";
import {MaxPool2D} from "@/app/ir/maxPool";

export enum Padding {Valid, Same}

export type Initializer = CustomInitializer | BuiltinInitializer

enum BuiltinInitializer {
  Zeroes
}

export class CustomInitializer {
}

export type Regularizer = BuiltinRegularizer

enum BuiltinRegularizer {
  None
}

export type ActivationF = BuiltinActivationF

enum BuiltinActivationF { Relu }

export namespace Graphs {
  export function mnist(): GraphNode[] {
    const defaultWeights: [Initializer, Regularizer] = [BuiltinInitializer.Zeroes, BuiltinRegularizer.None]
    const conv =
      new Conv2D(
        new Set(), 32n, Padding.Same, defaultWeights, null, new UUID("theInNode"), BuiltinActivationF.Relu, [28n, 28n], [2n, 2n]
      )
    const convGraph = new GraphNode(conv)

    const maxPool = new MaxPool2D(
      new Set(), convGraph.uniqueId, Padding.Same, [28n, 28n], [2n, 2n]
    )
    const maxPoolGraph = new GraphNode(maxPool)
    conv.outputs.add(maxPoolGraph.uniqueId)

    return [convGraph, maxPoolGraph]
  }
}
