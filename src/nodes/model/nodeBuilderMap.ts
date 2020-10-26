import { MlNode } from '@/app/ir/mainNodes';
import Conv2D from '@/app/ir/conv/Conv2D';
import MaxPool2D from '@/app/ir/maxPool/maxPool2D';
import Custom from '@/app/ir/Custom';

type Options = Map<string, any>
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  ['Convolution2D', Conv2D.build],
  ['MaxPooling2D', MaxPool2D.build],
  ['Custom', Custom.build],
]);
