import Custom from '@/app/ir/Custom';

import Adadelta from '@/app/ir/overview/optimizers/Adadelta';
import TrainClassifier from '@/app/ir/overview/train/TrainClassifier';
import NLLLoss from '@/app/ir/overview/loss/nllloss';
import Model from '@/app/ir/model/model';
import OverviewCustom from '@/app/ir/overview/OverviewCustom';
import InModel from './InModel';
import OutModel from './OutModel';
import Concat from './Concat';
import Conv2d from './model/conv2d';
import Conv1d from './model/conv1d';
import Conv3d from './model/conv3d';
import ConvTranspose1d from './model/convtranspose1d';
import ConvTranspose2d from './model/convtranspose2d';
import ConvTranspose3d from './model/convtranspose3d';
import MaxPool1d from './model/maxpool1d';
import MaxPool2d from './model/maxpool2d';
import MaxPool3d from './model/maxpool3d';

import OutData from './data/OutData';
import ToTensor from './data/ToTensor';
import Grayscale from './data/Grayscale';
import LoadCsv from './data/LoadCsv';
import LoadImages from './data/LoadImages';

export type MlNode = ModelNode | OverviewNode | DataNode
// MODEL NODES
export type Conv = Conv1d | Conv2d | Conv3d | ConvTranspose1d | ConvTranspose2d | ConvTranspose3d

export type MaxPool = MaxPool1d | MaxPool2d | MaxPool3d

export type ModelLayerNode = Conv | MaxPool

export type Operations = Concat

export type ModelNode = ModelLayerNode | InModel | OutModel | Custom | Operations

// OVERVIEW NODES
export type OverviewNode = OverviewCallableNode | OptimizerNode | LossNode | OverviewCustom
// callables have initCode and a callCode methods
export type OverviewCallableNode = TrainNode | OverviewCustom

// DATA NODES
export type DataNode = OutData | DataTransform | DataLoad

export type DataTransform = ToTensor | Grayscale

export type DataLoad = LoadCsv | LoadImages

// TRAIN NODES
export type OptimizerNode = Adadelta

export type LossNode = NLLLoss

export type TrainNode = TrainClassifier
