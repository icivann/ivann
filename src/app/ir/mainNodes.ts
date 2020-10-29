import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';
import Custom from '@/app/ir/Custom';
import InModel from './InModel';
import OutModel from './OutModel';
import Concat from './Concat';

export type MlNode = ModelNode

export type ModelLayerNode = Conv | MaxPool

export type Operations = Concat

export type ModelNode = ModelLayerNode | InModel | OutModel | Custom | Operations
