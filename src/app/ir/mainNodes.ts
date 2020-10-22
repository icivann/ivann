import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';
import InModel from './InModel';
import OutModel from './OutModel';

export type MlNode = ModelNode

export type ModelNode = Conv | MaxPool | InModel | OutModel
