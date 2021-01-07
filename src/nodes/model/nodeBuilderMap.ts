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

import ToTensor from '@/app/ir/data/ToTensor';
import Grayscale from '@/app/ir/data/Grayscale';
import OutData from '@/app/ir/data/OutData';
import Adadelta from '@/app/ir/overview/optimizers/Adadelta';
import TrainClassifier from '@/app/ir/overview/train/TrainClassifier';
import Model from '@/app/ir/model/model';
import Data from '@/app/ir/data/Data';

import Unfold from '@/app/ir/model/unfold';
import Fold from '@/app/ir/model/fold';
import MaxUnpool1d from '@/app/ir/model/maxunpool1d';
import MaxUnpool2d from '@/app/ir/model/maxunpool2d';
import MaxUnpool3d from '@/app/ir/model/maxunpool3d';
import AvgPool1d from '@/app/ir/model/avgpool1d';
import AvgPool2d from '@/app/ir/model/avgpool2d';
import AvgPool3d from '@/app/ir/model/avgpool3d';
import FractionalMaxPool2d from '@/app/ir/model/fractionalmaxpool2d';
import LPPool1d from '@/app/ir/model/lppool1d';
import LPPool2d from '@/app/ir/model/lppool2d';
import AdaptiveMaxPool1d from '@/app/ir/model/adaptivemaxpool1d';
import AdaptiveMaxPool2d from '@/app/ir/model/adaptivemaxpool2d';
import AdaptiveMaxPool3d from '@/app/ir/model/adaptivemaxpool3d';
import AdaptiveAvgPool1d from '@/app/ir/model/adaptiveavgpool1d';
import AdaptiveAvgPool2d from '@/app/ir/model/adaptiveavgpool2d';
import AdaptiveAvgPool3d from '@/app/ir/model/adaptiveavgpool3d';
import AlphaDropout from '@/app/ir/model/alphadropout';
import EmbeddingBag from '@/app/ir/model/embeddingbag';
import PairwiseDistance from '@/app/ir/model/pairwisedistance';
import CosineSimilarity from '@/app/ir/model/cosinesimilarity';
import ReflectionPad1d from '@/app/ir/model/reflectionpad1d';
import ReflectionPad2d from '@/app/ir/model/reflectionpad2d';
import ReplicationPad2d from '@/app/ir/model/replicationpad2d';
import ReplicationPad3d from '@/app/ir/model/replicationpad3d';
import ReplicationPad1d from '@/app/ir/model/replicationpad1d';
import ZeroPad2d from '@/app/ir/model/zeropad2d';
import ConstantPad1d from '@/app/ir/model/constantpad1d';
import ConstantPad2d from '@/app/ir/model/constantpad2d';
import ConstantPad3d from '@/app/ir/model/constantpad3d';
import ELU from '@/app/ir/model/elu';
import Hardshrink from '@/app/ir/model/hardshrink';
import Hardsigmoid from '@/app/ir/model/hardsigmoid';
import MultiheadAttention from '@/app/ir/model/multiheadattention';
import PReLU from '@/app/ir/model/prelu';
import LeakyReLU from '@/app/ir/model/leakyrelu';
import Hardswish from '@/app/ir/model/hardswish';
import Hardtanh from '@/app/ir/model/hardtanh';
import ReLU6 from '@/app/ir/model/relu6';
import RReLU from '@/app/ir/model/rrelu';
import CELU from '@/app/ir/model/celu';
import GELU from '@/app/ir/model/gelu';
import SELU from '@/app/ir/model/selu';
import Sigmoid from '@/app/ir/model/sigmoid';
import SiLU from '@/app/ir/model/silu';
import Softplus from '@/app/ir/model/softplus';
import Softshrink from '@/app/ir/model/softshrink';
import Softsign from '@/app/ir/model/softsign';
import Tanh from '@/app/ir/model/tanh';
import LogSigmoid from '@/app/ir/model/logsigmoid';
import Tanhshrink from '@/app/ir/model/tanhshrink';
import Threshold from '@/app/ir/model/threshold';
import L1Loss from '@/app/ir/model/l1loss';
import MSELoss from '@/app/ir/model/mseloss';
import CrossEntropyLoss from '@/app/ir/model/crossentropyloss';
import CTCLoss from '@/app/ir/model/ctcloss';
import NLLLoss from '@/app/ir/model/nllloss';
import PoissonNLLLoss from '@/app/ir/model/poissonnllloss';
import KLDivLoss from '@/app/ir/model/kldivloss';
import BCELoss from '@/app/ir/model/bceloss';
import BCEWithLogitsLoss from '@/app/ir/model/bcewithlogitsloss';
import MarginRankingLoss from '@/app/ir/model/marginrankingloss';
import HingeEmbeddingLoss from '@/app/ir/model/hingeembeddingloss';
import MultiLabelMarginLoss from '@/app/ir/model/multilabelmarginloss';
import SmoothL1Loss from '@/app/ir/model/smoothl1loss';
import MultiLabelSoftMarginLoss from '@/app/ir/model/multilabelsoftmarginloss';
import CosineEmbeddingLoss from '@/app/ir/model/cosineembeddingloss';
import MultiMarginLoss from '@/app/ir/model/multimarginloss';
import TripletMarginLoss from '@/app/ir/model/tripletmarginloss';
import LogSoftmax from '@/app/ir/model/logsoftmax';
import Softmax2d from '@/app/ir/model/softmax2d';
import AdaptiveLogSoftmaxWithLoss from '@/app/ir/model/adaptivelogsoftmaxwithloss';
import BatchNorm1d from '@/app/ir/model/batchnorm1d';
import BatchNorm2d from '@/app/ir/model/batchnorm2d';
import BatchNorm3d from '@/app/ir/model/batchnorm3d';
import GroupNorm from '@/app/ir/model/groupnorm';
import SyncBatchNorm from '@/app/ir/model/syncbatchnorm';
import InstanceNorm1d from '@/app/ir/model/instancenorm1d';
import InstanceNorm2d from '@/app/ir/model/instancenorm2d';
import InstanceNorm3d from '@/app/ir/model/instancenorm3d';
import LocalResponseNorm from '@/app/ir/model/localresponsenorm';
import TransformerEncoderLayer from '@/app/ir/model/transformerencoderlayer';
import TransformerDecoderLayer from '@/app/ir/model/transformerdecoderlayer';
import LoadCsv from '@/app/ir/data/LoadCsv';
import LoadImages from '@/app/ir/data/LoadImages';
import RNNBase from '@/app/ir/model/rnnbase';
import RNN from '@/app/ir/model/rnn';
import LSTM from '@/app/ir/model/lstm';
import GRU from '@/app/ir/model/gru';
import RNNCell from '@/app/ir/model/rnncell';
import LSTMCell from '@/app/ir/model/lstmcell';
import GRUCell from '@/app/ir/model/grucell';
import Adamax from '@/app/ir/overview/optimizers/adamax';
import SparseAdam from '@/app/ir/overview/optimizers/sparseadam';
import AdamW from '@/app/ir/overview/optimizers/adamw';
import LBFGS from '@/app/ir/overview/optimizers/lbfgs';
import ASGD from '@/app/ir/overview/optimizers/asgd';
import RMSprop from '@/app/ir/overview/optimizers/rmsprop';
import Rprop from '@/app/ir/overview/optimizers/rprop';
import SGD from '@/app/ir/overview/optimizers/sgd';
import Adagrad from '@/app/ir/overview/optimizers/adagrad';
import Adam from '@/app/ir/overview/optimizers/adam';

type Options = Map<string, any>
// eslint-disable-next-line import/prefer-default-export
export const nodeBuilder: Map<string, (r: Options) => MlNode> = new Map([
  ['ModelNode', Model.build],
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
  // transformer
  ['Transformer', Transformer.build],
  ['TransformerEncoderLayer', TransformerEncoderLayer.build],
  ['TransformerDecoderLayer', TransformerDecoderLayer.build],
  ['Linear', Linear.build],
  ['Bilinear', Bilinear.build],
  ['Softmin', Softmin.build],
  ['Softmax', Softmax.build],
  // Data
  ['OutData', OutData.build],
  ['ToTensor', ToTensor.build],
  ['Grayscale', Grayscale.build],
  ['Transformer', Transformer.build],
  ['DataNode', Data.build],
  ['LoadCsv', LoadCsv.build],
  ['LoadImages', LoadImages.build],
  // Optimizers
  ['Adadelta', Adadelta.build],
  // Training
  ['TrainClassifier', TrainClassifier.build],
  ['Unfold', Unfold.build],
  ['Fold', Fold.build],
  ['MaxUnpool1d', MaxUnpool1d.build],
  ['MaxUnpool2d', MaxUnpool2d.build],
  ['MaxUnpool3d', MaxUnpool3d.build],
  ['AvgPool1d', AvgPool1d.build],
  ['AvgPool2d', AvgPool2d.build],
  ['AvgPool3d', AvgPool3d.build],
  ['FractionalMaxPool2d', FractionalMaxPool2d.build],
  ['LPPool1d', LPPool1d.build],
  ['LPPool2d', LPPool2d.build],
  ['AdaptiveMaxPool1d', AdaptiveMaxPool1d.build],
  ['AdaptiveMaxPool2d', AdaptiveMaxPool2d.build],
  ['AdaptiveMaxPool3d', AdaptiveMaxPool3d.build],
  ['AdaptiveAvgPool1d', AdaptiveAvgPool1d.build],
  ['AdaptiveAvgPool2d', AdaptiveAvgPool2d.build],
  ['AdaptiveAvgPool3d', AdaptiveAvgPool3d.build],
  ['AlphaDropout', AlphaDropout.build],
  ['EmbeddingBag', EmbeddingBag.build],
  ['CosineSimilarity', CosineSimilarity.build],
  ['PairwiseDistance', PairwiseDistance.build],
  // padding
  ['ReflectionPad1d', ReflectionPad1d.build],
  ['ReflectionPad2d', ReflectionPad2d.build],
  ['ReplicationPad1d', ReplicationPad1d.build],
  ['ReplicationPad2d', ReplicationPad2d.build],
  ['ReplicationPad3d', ReplicationPad3d.build],
  ['ZeroPad2d', ZeroPad2d.build],
  ['ConstantPad1d', ConstantPad1d.build],
  ['ConstantPad2d', ConstantPad2d.build],
  ['ConstantPad3d', ConstantPad3d.build],
  // activations
  ['ELU', ELU.build],
  ['Hardshrink', Hardshrink.build],
  ['Hardsigmoid', Hardsigmoid.build],
  ['Hardtanh', Hardtanh.build],
  ['Hardswish', Hardswish.build],
  ['LeakyReLU', LeakyReLU.build],
  ['MultiheadAttention', MultiheadAttention.build],
  ['PReLU', PReLU.build],
  ['ReLU6', ReLU6.build],
  ['RReLU', RReLU.build],
  ['SELU', SELU.build],
  ['CELU', CELU.build],
  ['GELU', GELU.build],
  ['Sigmoid', Sigmoid.build],
  ['SiLU', SiLU.build],
  ['Softplus', Softplus.build],
  ['Softshrink', Softshrink.build],
  ['Softsign', Softsign.build],
  ['Tanh', Tanh.build],
  ['LogSigmoid', LogSigmoid.build],
  ['Tanhshrink', Tanhshrink.build],
  ['Threshold', Threshold.build],
  // loss functions
  ['L1Loss', L1Loss.build],
  ['MSELoss', MSELoss.build],
  ['CrossEntropyLoss', CrossEntropyLoss.build],
  ['CTCLoss', CTCLoss.build],
  ['NLLLoss', NLLLoss.build],
  ['PoissonNLLLoss', PoissonNLLLoss.build],
  ['KLDivLoss', KLDivLoss.build],
  ['BCELoss', BCELoss.build],
  ['BCEWithLogitsLoss', BCEWithLogitsLoss.build],
  ['MarginRankingLoss', MarginRankingLoss.build],
  ['HingeEmbeddingLoss', HingeEmbeddingLoss.build],
  ['MultiLabelMarginLoss', MultiLabelMarginLoss.build],
  ['SmoothL1Loss', SmoothL1Loss.build],
  ['MultiLabelSoftMarginLoss', MultiLabelSoftMarginLoss.build],
  ['CosineEmbeddingLoss', CosineEmbeddingLoss.build],
  ['MultiMarginLoss', MultiMarginLoss.build],
  ['TripletMarginLoss', TripletMarginLoss.build],
  ['LogSoftmax', LogSoftmax.build],
  ['Softmax2d', Softmax2d.build],
  ['AdaptiveLogSoftmaxWithLoss', AdaptiveLogSoftmaxWithLoss.build],
  // normalise
  ['BatchNorm1d', BatchNorm1d.build],
  ['BatchNorm2d', BatchNorm2d.build],
  ['BatchNorm3d', BatchNorm3d.build],
  ['GroupNorm', GroupNorm.build],
  ['SyncBatchNorm', SyncBatchNorm.build],
  ['InstanceNorm1d', InstanceNorm1d.build],
  ['InstanceNorm2d', InstanceNorm2d.build],
  ['InstanceNorm3d', InstanceNorm3d.build],
  ['LocalResponseNorm', LocalResponseNorm.build],

  ['RNNBase', RNNBase.build],
  ['RNN', RNN.build],
  ['LSTM', LSTM.build],
  ['GRU', GRU.build],
  ['RNNCell', RNNCell.build],
  ['LSTMCell', LSTMCell.build],
  ['GRUCell', GRUCell.build],

  ['Adamax', Adamax.build],
  ['SparseAdam', SparseAdam.build],
  ['AdamW', AdamW.build],
  ['Adam', Adam.build],
  ['Adagrad', Adagrad.build],
  ['ASGD', ASGD.build],
  ['LBFGS', LBFGS.build],
  ['RMSprop', RMSprop.build],
  ['Rprop', Rprop.build],
  ['SGD', SGD.build],

]);
