import { UUID } from '@/app/util';

export type MlNode = ModelNode

export abstract class ModelNode {
  public abstract code(): string
}

export type InOutNode = InNode & OutNode

export interface InNode {
  outputs: Set<UUID>;
}

export interface OutNode {
  input: UUID;
}

export interface Out2Node {
  input: [UUID, UUID];
}
