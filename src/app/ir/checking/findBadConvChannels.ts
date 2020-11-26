import Graph from '@/app/ir/Graph';
import Conv1d from '@/app/ir/model/conv1d';
import Conv2d from '@/app/ir/model/conv2d';
import Conv3d from '@/app/ir/model/conv3d';
import GraphNode from '@/app/ir/GraphNode';
import IrError from '@/app/ir/checking/irError';
import { Severity } from '@/app/ir/checking/severity';
import ConvTranspose1d from '@/app/ir/model/convtranspose1d';
import ConvTranspose2d from '@/app/ir/model/convtranspose2d';
import ConvTranspose3d from '@/app/ir/model/convtranspose3d';

/**
 * Finds all nodes in the graph that are right before a Conv node and are Conv
 * nodes themselves, and checks their channels match.
 */
export const findBadConvChannels = (graph: Graph) => {
  type Conv = Conv1d | Conv2d | Conv3d;
  const isConv = (node: GraphNode) => node.mlNode instanceof Conv1d
    || node.mlNode instanceof Conv2d
    || node.mlNode instanceof Conv3d
    || node.mlNode instanceof ConvTranspose1d
    || node.mlNode instanceof ConvTranspose2d
    || node.mlNode instanceof ConvTranspose3d;
  return graph.nodesAsArray
    .filter(isConv)
    .flatMap((current) => {
      const currentConv = current.mlNode as Conv;
      return graph.nextNodesFrom(current)
        .filter(isConv)
        .map((n) => [n, n.mlNode] as [GraphNode, Conv])
        .filter(([_, conv]) => conv.in_channels !== currentConv.out_channels)
        .map(([offenderNode, offenderConv]) => {
          const nodes = `${currentConv.name} and ${offenderConv.name}`;
          const channels = `${currentConv.out_channels} and ${offenderConv.in_channels}`;
          return new IrError(
            [offenderNode, current],
            Severity.Error,
            `Expected channels of convolutional layers ${nodes} to match, but instead got ${channels}`,
          );
        });
    });
};
