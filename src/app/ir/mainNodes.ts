import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';
import Dense from '@/app/ir/dense';
import Dropout from '@/app/ir/dropout';

export type MlNode = ModelNode

export type ModelNode = Dense | Dropout | Conv | MaxPool
