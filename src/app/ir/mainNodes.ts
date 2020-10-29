import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';
import Custom from '@/app/ir/Custom';
import InModel from '@/app/ir/InModel';
import OutModel from '@/app/ir/OutModel';

export type MlNode = ModelNode

export type ModelNode = Conv | MaxPool | Custom | InModel | OutModel
