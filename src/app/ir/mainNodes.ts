import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';

export type MlNode = ModelNode

export type ModelNode = Conv | MaxPool
