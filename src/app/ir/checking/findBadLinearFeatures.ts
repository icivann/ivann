import Linear from '@/app/ir/model/linear';
import { adjacencyCheck } from '@/app/ir/checking/adjacencyCheck';

/**
 * Finds all nodes in the graph that are right before a Conv node and are Conv
 * nodes themselves, and checks their channels match.
 */

export const findBadLinearFeatures = adjacencyCheck(
  (node) => node.mlNode instanceof Linear,
  (fst, snd) => (fst.mlNode as Linear).out_features === (snd.mlNode as Linear).in_features,
  (fst, snd) => `Expected features of linear layers ${fst.mlNode.name} and ${snd.mlNode.name} to match `
    + `but instead got ${(fst.mlNode as Linear).out_features} and ${(snd.mlNode as Linear).in_features}`,
);
