import { MlNode } from '@/app/ir/mainNodes';
import Conv2D from '@/app/ir/conv/Conv2D';
import MaxPool2D from '@/app/ir/maxPool/maxPool2D';
import Custom from '@/app/ir/Custom';
import InModel from '@/app/ir/InModel';
import OutModel from '@/app/ir/OutModel';
import Concat from '@/app/ir/Concat';
import Conv1d from '@/app/ir/pytorch model/conv1d';
import ConvTranspose1d from '@/app/ir/pytorch model/convtranspose1d';
import ConvTranspose2d from '@/app/ir/pytorch model/convtranspose2d';
import ConvTranspose3d from '@/app/ir/pytorch model/convtranspose3d';
import MaxPool1d from '@/app/ir/pytorch model/maxpool1d';
import MaxPool2d from '@/app/ir/pytorch model/maxpool2d';
import MaxPool3d from '@/app/ir/pytorch model/maxpool3d';

type Options = Map<string, any>
// eslint-disable-next-line import/prefer-default-export
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  ['Convolution1D', Conv1d.build],
  ['Convolution2D', Conv1d.build],
  ['Convolution3D', Conv1d.build],
  ['Convolution1D', Conv1d.build],
  ['ConvolutionTranspose1d', ConvTranspose1d.build],
  ['ConvolutionTranspose2d', ConvTranspose2d.build],
  ['ConvolutionTranspose3d', ConvTranspose3d.build],
  ['MaxPooling1D', MaxPool1d.build],
  ['MaxPooling2D', MaxPool2d.build],
  ['MaxPooling3D', MaxPool3d.build],
  ['Custom', Custom.build as (r: Options) => MlNode],
  ['Concat', Concat.build],
  ['InModel', InModel.build],
  ['OutModel', OutModel.build],
]);
