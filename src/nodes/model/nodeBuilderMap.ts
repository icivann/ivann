import { MlNode } from '@/app/ir/mainNodes';
import Conv2D from '@/app/ir/conv/Conv2D';
import MaxPool2D from '@/app/ir/maxPool/maxPool2D';
import Custom from '@/app/ir/Custom';
import InModel from '@/app/ir/InModel';
import OutModel from '@/app/ir/OutModel';
import Concat from '@/app/ir/Concat';

type Options = Map<string, any>
// eslint-disable-next-line import/prefer-default-export
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  ['Convolution2D', Conv2D.build],
  ['MaxPooling2D', MaxPool2D.build],
  ['Custom', Custom.build as (r: Options) => MlNode],
  ['Concat', Concat.build],
  ['InModel', InModel.build],
  ['OutModel', OutModel.build],
]);
