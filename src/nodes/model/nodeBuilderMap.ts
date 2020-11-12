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
import Dropout from '@/app/ir/model/dropout';
import Dropout2d from '@/app/ir/model/dropout2d';
import Dropout3d from '@/app/ir/model/dropout3d';
import ReLU from '@/app/ir/model/relu';
import Conv2d from '@/app/ir/model/conv2d';
import Conv3d from '@/app/ir/model/conv3d';
import Transformer from '@/app/ir/model/transformer';
import Linear from '@/app/ir/model/linear';
import Bilinear from '@/app/ir/model/bilinear';
import Softmin from '@/app/ir/model/softmin';
import Softmax from '@/app/ir/model/softmax';

type Options = Map<string, any>
// eslint-disable-next-line import/prefer-default-export
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  ['Conv1d', Conv1d.build],
  ['Conv2d', Conv2d.build],
  ['Conv3d', Conv3d.build],
  ['ConvTranspose1d', ConvTranspose1d.build],
  ['ConvTranspose2d', ConvTranspose2d.build],
  ['ConvTranspose3d', ConvTranspose3d.build],
  ['MaxPool1d', MaxPool1d.build],
  ['MaxPool2d', MaxPool2d.build],
  ['MaxPool3d', MaxPool3d.build],
  ['Dropout', Dropout.build],
  ['Dropout2d', Dropout2d.build],
  ['Dropout3d', Dropout3d.build],
  ['Relu', ReLU.build],
  ['Custom', Custom.build as (r: Options) => MlNode],
  ['Concat', Concat.build],
  ['InModel', InModel.build],
  ['OutModel', OutModel.build],
  ['Transformer', Transformer.build],
  ['Linear', Linear.build],
  ['Bilinear', Bilinear.build],
  ['Softmin', Softmin.build],
  ['Softmax', Softmax.build],
]);
