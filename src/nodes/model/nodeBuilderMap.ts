import { MlNode } from '@/app/ir/mainNodes';
import Conv2D from '@/app/ir/conv/Conv2D';
import MaxPool2D from '@/app/ir/maxPool/maxPool2D';
import Custom from '@/app/ir/Custom';
import InModel from '@/app/ir/InModel';
import OutModel from '@/app/ir/OutModel';
import Concat from '@/app/ir/Concat';
import Conv1d from '@/app/ir/pytorch model/conv1d';

type Options = Map<string, any>
// eslint-disable-next-line import/prefer-default-export
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  ['Convolution1D', Conv1d.build],
  ['Convolution2D', Conv1d.build],
  ['Convolution3D', Conv1d.build],
  ['Convolution1D', Conv1d.build],
  ['Custom', Custom.build as (r: Options) => MlNode],
  ['Concat', Concat.build],
  ['InModel', InModel.build],
  ['OutModel', OutModel.build],
]);
