import Graph from '@/app/ir/Graph';
import TrainClassifier from '@/app/ir/overview/train/TrainClassifier';
import { MlNode } from '@/app/ir/mainNodes';
import OverviewCustom from '@/nodes/overview/OverviewCustom';
import { Severity } from '@/app/ir/checking/severity';
import IrError from '@/app/ir/checking/irError';

const validOptimiser = (node: MlNode) => node instanceof OverviewCustom;
// || node instanceof Optimiser

export const findNonOptimizersInTraining = (graph: Graph) => graph.nodesAsArray
  .filter((n) => n.mlNode instanceof TrainClassifier)
  .flatMap((fst) => {
    const optimiser = graph.prevNodeFrom(fst, 'Optimiser');
    if (optimiser === undefined || validOptimiser(optimiser.mlNode)) return [];
    return [fst, optimiser].map((_) => new IrError(
      [fst, optimiser],
      Severity.Warning,
      `Node ${optimiser.mlNode.name} used as Optimiser in training does not look `
        + 'like a valid optimiser. Use an Optimiser node or a Custom node',
    ));
  });
