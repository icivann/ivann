import { ModelNode } from '@/app/ir/mainNodes';
import Conv2D from '@/app/ir/conv/Conv2D';
import MaxPool2D from '@/app/ir/maxPool/maxPool2D';
import Dense from '@/app/ir/dense';
import Conv1D from '@/app/ir/conv/Conv1D';
import Dropout from '@/app/ir/dropout';

type Options = Map<string, any>
export const nodeBuilder: Map<string, (r: Options) => ModelNode> = new Map([
  ['Convolution2D', Conv2D.build],
  ['MaxPooling2D', MaxPool2D.build],
  ['Convolution1D', Conv1D.build],
  ['Dropout', Dropout.build],
  ['Dense', Dense.build],
]);
