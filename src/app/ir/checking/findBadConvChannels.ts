import Conv1d from '@/app/ir/model/conv1d';
import Conv2d from '@/app/ir/model/conv2d';
import Conv3d from '@/app/ir/model/conv3d';
import ConvTranspose1d from '@/app/ir/model/convtranspose1d';
import ConvTranspose2d from '@/app/ir/model/convtranspose2d';
import ConvTranspose3d from '@/app/ir/model/convtranspose3d';
import { adjacencyCheck } from '@/app/ir/checking/adjacencyCheck';

type Conv = Conv1d | Conv2d | Conv3d | ConvTranspose1d | ConvTranspose2d | ConvTranspose3d;

/**
 * Finds all nodes in the graph that are right before a Conv node and are Conv
 * nodes themselves, and checks their channels match.
 */

export const findBadConvChannels = adjacencyCheck(
  (node) => node.mlNode instanceof Conv1d
    || node.mlNode instanceof Conv2d
    || node.mlNode instanceof Conv3d
    || node.mlNode instanceof ConvTranspose1d
    || node.mlNode instanceof ConvTranspose2d
    || node.mlNode instanceof ConvTranspose3d,
  (fst, snd) => (fst.mlNode as Conv).out_channels === (snd.mlNode as Conv).in_channels,
  (fst, snd) => `Expected channels of convolutional layers ${fst.mlNode.name} and ${snd.mlNode.name} to match `
    + `but instead got ${(fst.mlNode as Conv).out_channels} and ${(snd.mlNode as Conv).in_channels}`,
);
