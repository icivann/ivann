import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';
import InModel from './InModel';
import OutModel from './OutModel';

export type ModelLayerNode = Conv | MaxPool

export type ModelNode = ModelLayerNode | InModel | OutModel
