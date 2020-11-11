import Graph from '@/app/ir/Graph';
import IrError from '@/app/ir/checking/irError';
import { Severity } from '@/app/ir/checking/severity';

export const findDanglingInterfaces = (graph: Graph) => graph.nodesAsArray
  // get nodes with output interfaces
  .filter((n) => n.danglingInterfaces.length != 0)
  .flatMap((node) => node.danglingInterfaces.map((name) => new IrError(
    [node],
    Severity.Error,
    `Node missing a connection at '${name} in ${typeof node.mlNode}`,
  )));
