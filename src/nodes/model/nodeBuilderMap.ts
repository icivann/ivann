import { MlNode } from '@/app/ir/mainNodes';
import Custom from '@/app/ir/Custom';
import InModel from '@/app/ir/InModel';
import OutModel from '@/app/ir/OutModel';
import Concat from '@/app/ir/Concat';
import Conv1d from '@/app/ir/model/conv1d';
import ConvTranspose1d from '@/app/ir/model/convtranspose1d';
import ConvTranspose2d from '@/app/ir/model/convtranspose2d';
import ConvTranspose3d from '@/app/ir/model/convtranspose3d';
import MaxPool1d from '@/app/ir/model/maxpool1d';
import MaxPool2d from '@/app/ir/model/maxpool2d';
import MaxPool3d from '@/app/ir/model/maxpool3d';

type Options = Map<string, any>
// eslint-disable-next-line import/prefer-default-export
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  ['Conv1D', Conv1d.build],
  ['Conv2D', Conv1d.build],
  ['Conv3D', Conv1d.build],
  ['ConvTranspose1d', ConvTranspose1d.build],
  ['ConvTranspose2d', ConvTranspose2d.build],
  ['ConvTranspose3d', ConvTranspose3d.build],
  ['MaxPooling1D', MaxPool1d.build],
  ['MaxPooling2D', MaxPool2d.build],
  ['MaxPooling3D', MaxPool3d.build],
  ['Custom', Custom.build as (r: Options) => MlNode],
  ['Concat', Concat.build],
  ['InModel', InModel.build],
  ['OutModel', OutModel.build],
]);
