import GraphNode from '@/app/ir/GraphNode';
import { Severity } from '@/app/ir/checking/severity';
import Graph from '@/app/ir/Graph';
import IrError from '@/app/ir/checking/irError';

/**
 * A function that abstracts the common pattern of performing a check when
 * two nodes happen to be adjacent and both meet a condition
 */
export function adjacencyCheck(
  nodesToCheck: (node: GraphNode) => boolean,
  nodesComplyIf: (fst: GraphNode, snd: GraphNode) => boolean,
  message: (fstOffender: GraphNode, sndOffender: GraphNode) => string,
  severity: Severity = Severity.Error,
): (graph: Graph) => IrError[] {
  return (graph: Graph) => graph.nodesAsArray
    .filter(nodesToCheck)
    .flatMap((fst) => graph.nextNodesFrom(fst)
      .filter(nodesToCheck)
      .filter((snd) => !nodesComplyIf(fst, snd))
      .map((snd) => new IrError(
        [fst, snd],
        severity,
        message(fst, snd),
      )));
}
