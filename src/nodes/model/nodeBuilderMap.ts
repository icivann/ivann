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
import Adadelta from '@/app/ir/optimizers/Adadelta';
import TrainClassifier from '@/app/ir/train/TrainClassifier';
import Model from '@/app/ir/model/model';

type Options = Map<string, any>
// eslint-disable-next-line import/prefer-default-export
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  // Model Nodes
  ['ModelNode', Model.build],
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
  // Optimizers
  ['Adadelta', Adadelta.build],
  // Training
  ['TrainClassifier', TrainClassifier.build],
]);
