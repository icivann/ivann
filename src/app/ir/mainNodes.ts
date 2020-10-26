import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';
import Custom from '@/app/ir/Custom';

export type MlNode = ModelNode

export type ModelNode = Conv | MaxPool | Custom
