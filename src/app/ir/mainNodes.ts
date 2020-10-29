import { Conv } from '@/app/ir/conv/Conv';
import { MaxPool } from '@/app/ir/maxPool/maxPool';
import Custom from '@/app/ir/Custom';
import Dropout from '@/app/ir/dropout';
import Dense from '@/app/ir/dense';

export type MlNode = ModelNode

export type ModelNode = Dense | Dropout | Conv | MaxPool | Custom
