import AbstractCanvas from '@/components/canvas/AbstractCanvas';
import { ModelCategories, ModelNodes } from '@/nodes/model/Types';

import Conv1d from '@/nodes/model/Conv1d';
import Conv2d from '@/nodes/model/Conv2d';
import Conv3d from '@/nodes/model/Conv3d';
import OutModel from '@/nodes/model/OutModel';
import Concat from '@/nodes/model/Concat';
import InModel from '@/nodes/model/InModel';
import Convtranspose1d from '@/nodes/model/Convtranspose1d';
import Convtranspose2d from '@/nodes/model/Convtranspose2d';
import Convtranspose3d from '@/nodes/model/Convtranspose3d';
import Maxpool1d from '@/nodes/model/Maxpool1d';
import Maxpool2d from '@/nodes/model/Maxpool2d';
import Maxpool3d from '@/nodes/model/Maxpool3d';
import Dropout from '@/nodes/model/Dropout';
import Dropout2d from '@/nodes/model/Dropout2d';
import Dropout3d from '@/nodes/model/Dropout3d';
import Transformer from '@/nodes/model/Transformer';
import Softmin from '@/nodes/model/Softmin';
import Softmax from '@/nodes/model/Softmax';
import Flatten from '@/nodes/model/Flatten';
import Bilinear from '@/nodes/model/Bilinear';
import Linear from '@/nodes/model/Linear';
import { Editor } from '@baklavajs/core';
import Unfold from '@/nodes/model/Unfold';
import Fold from '@/nodes/model/Fold';
import MaxUnpool1d from '@/nodes/model/Maxunpool1d';
import MaxUnpool2d from '@/nodes/model/Maxunpool2d';
import MaxUnpool3d from '@/nodes/model/Maxunpool3d';
import AvgPool1d from '@/nodes/model/Avgpool1d';
import AvgPool2d from '@/nodes/model/Avgpool2d';
import AvgPool3d from '@/nodes/model/Avgpool3d';
import FractionalMaxPool2d from '@/nodes/model/Fractionalmaxpool2d';
import LPPool1d from '@/nodes/model/Lppool1d';
import LPPool2d from '@/nodes/model/Lppool2d';
import AdaptiveMaxPool1d from '@/nodes/model/Adaptivemaxpool1d';
import AdaptiveMaxPool2d from '@/nodes/model/Adaptivemaxpool2d';
import AdaptiveMaxPool3d from '@/nodes/model/Adaptivemaxpool3d';
import AdaptiveAvgPool1d from '@/nodes/model/Adaptiveavgpool1d';
import AdaptiveAvgPool3d from '@/nodes/model/Adaptiveavgpool3d';
import AdaptiveAvgPool2d from '@/nodes/model/Adaptiveavgpool2d';
import AlphaDropout from '@/nodes/model/Alphadropout';
import CosineSimilarity from '@/nodes/model/Cosinesimilarity';
import PairwiseDistance from '@/nodes/model/Pairwisedistance';
import ReflectionPad1d from '@/nodes/model/Reflectionpad1d';
import ReflectionPad2d from '@/nodes/model/Reflectionpad2d';
import ReplicationPad1d from '@/nodes/model/Replicationpad1d';
import ReplicationPad2d from '@/nodes/model/Replicationpad2d';
import ReplicationPad3d from '@/nodes/model/Replicationpad3d';
import ZeroPad2d from '@/nodes/model/Zeropad2d';
import ConstantPad1d from '@/nodes/model/Constantpad1d';
import ConstantPad2d from '@/nodes/model/Constantpad2d';
import ConstantPad3d from '@/nodes/model/Constantpad3d';
import ELU from '@/nodes/model/Elu';
import Hardshrink from '@/nodes/model/Hardshrink';
import Hardsigmoid from '@/nodes/model/Hardsigmoid';
import Hardtanh from '@/nodes/model/Hardtanh';
import Hardswish from '@/nodes/model/Hardswish';
import LeakyReLU from '@/nodes/model/Leakyrelu';
import MultiheadAttention from '@/nodes/model/Multiheadattention';
import PReLU from '@/nodes/model/Prelu';
import ReLU6 from '@/nodes/model/Relu6';
import RReLU from '@/nodes/model/Rrelu';
import SELU from '@/nodes/model/Selu';
import CELU from '@/nodes/model/Celu';
import GELU from '@/nodes/model/Gelu';
import Sigmoid from '@/nodes/model/Sigmoid';
import SiLU from '@/nodes/model/Silu';
import Softplus from '@/nodes/model/Softplus';
import Softshrink from '@/nodes/model/Softshrink';
import Softsign from '@/nodes/model/Softsign';
import Tanh from '@/nodes/model/Tanh';
import LogSigmoid from '@/nodes/model/Logsigmoid';
import Tanhshrink from '@/nodes/model/Tanhshrink';
import Threshold from '@/nodes/model/Threshold';
import ReLU from '@/nodes/model/Relu';
import LogSoftmax from '@/nodes/model/Logsoftmax';
import Softmax2d from '@/nodes/model/Softmax2d';
import AdaptiveLogSoftmaxWithLoss from '@/nodes/model/Adaptivelogsoftmaxwithloss';
import BatchNorm1d from '@/nodes/model/Batchnorm1d';
import BatchNorm2d from '@/nodes/model/Batchnorm2d';
import BatchNorm3d from '@/nodes/model/Batchnorm3d';
import GroupNorm from '@/nodes/model/Groupnorm';
import SyncBatchNorm from '@/nodes/model/Syncbatchnorm';
import InstanceNorm1d from '@/nodes/model/Instancenorm1d';
import InstanceNorm2d from '@/nodes/model/Instancenorm2d';
import InstanceNorm3d from '@/nodes/model/Instancenorm3d';
import LocalResponseNorm from '@/nodes/model/Localresponsenorm';
import TransformerEncoderLayer from '@/nodes/model/Transformerencoderlayer';
import TransformerDecoderLayer from '@/nodes/model/Transformerdecoderlayer';
import RNNBase from '@/nodes/model/Rnnbase';
import RNN from '@/nodes/model/Rnn';
import LSTM from '@/nodes/model/Lstm';
import GRU from '@/nodes/model/Gru';
import RNNCell from '@/nodes/model/Rnncell';
import LSTMCell from '@/nodes/model/Lstmcell';
import GRUCell from '@/nodes/model/Grucell';
import EmbeddingBag from '@/nodes/model/Embeddingbag';
import Embedding from '@/nodes/model/Embedding';
import ModelCustom from '@/nodes/model/ModelCustom';

export default class ModelCanvas extends AbstractCanvas {
  public nodeList = [
    {
      category: ModelCategories.Conv,
      nodes: [
        {
          name: ModelNodes.Conv1d,
          node: Conv1d,
        },
        {
          name: ModelNodes.Conv2d,
          node: Conv2d,
        },
        {
          name: ModelNodes.Conv3d,
          node: Conv3d,
        },
        {
          name: ModelNodes.ConvTranspose1d,
          node: Convtranspose1d,
        },
        {
          name: ModelNodes.ConvTranspose2d,
          node: Convtranspose2d,
        },
        {
          name: ModelNodes.ConvTranspose3d,
          node: Convtranspose3d,
        },
        {
          name: ModelNodes.Unfold,
          node: Unfold,
        },

        {
          name: ModelNodes.Fold,
          node: Fold,
        },
      ],
    },
    {
      category: ModelCategories.Pool,
      nodes: [
        {
          name: ModelNodes.MaxPool1d,
          node: Maxpool1d,
        },
        {
          name: ModelNodes.MaxPool2d,
          node: Maxpool2d,
        },
        {
          name: ModelNodes.MaxPool3d,
          node: Maxpool3d,
        },
        {
          name: ModelNodes.MaxUnpool1d,
          node: MaxUnpool1d,
        },

        {
          name: ModelNodes.MaxUnpool2d,
          node: MaxUnpool2d,
        },

        {
          name: ModelNodes.MaxUnpool3d,
          node: MaxUnpool3d,
        },

        {
          name: ModelNodes.AvgPool1d,
          node: AvgPool1d,
        },

        {
          name: ModelNodes.AvgPool2d,
          node: AvgPool2d,
        },

        {
          name: ModelNodes.AvgPool3d,
          node: AvgPool3d,
        },

        {
          name: ModelNodes.FractionalMaxPool2d,
          node: FractionalMaxPool2d,
        },

        {
          name: ModelNodes.LPPool1d,
          node: LPPool1d,
        },

        {
          name: ModelNodes.LPPool2d,
          node: LPPool2d,
        },

        {
          name: ModelNodes.AdaptiveMaxPool1d,
          node: AdaptiveMaxPool1d,
        },

        {
          name: ModelNodes.AdaptiveMaxPool2d,
          node: AdaptiveMaxPool2d,
        },

        {
          name: ModelNodes.AdaptiveMaxPool3d,
          node: AdaptiveMaxPool3d,
        },

        {
          name: ModelNodes.AdaptiveAvgPool1d,
          node: AdaptiveAvgPool1d,
        },

        {
          name: ModelNodes.AdaptiveAvgPool2d,
          node: AdaptiveAvgPool2d,
        },

        {
          name: ModelNodes.AdaptiveAvgPool3d,
          node: AdaptiveAvgPool3d,
        },
      ],
    },
    {
      category: ModelCategories.Padding,
      nodes: [

        {
          name: ModelNodes.ReflectionPad1d,
          node: ReflectionPad1d,
        },

        {
          name: ModelNodes.ReflectionPad2d,
          node: ReflectionPad2d,
        },

        {
          name: ModelNodes.ReplicationPad1d,
          node: ReplicationPad1d,
        },

        {
          name: ModelNodes.ReplicationPad2d,
          node: ReplicationPad2d,
        },

        {
          name: ModelNodes.ReplicationPad3d,
          node: ReplicationPad3d,
        },

        {
          name: ModelNodes.ZeroPad2d,
          node: ZeroPad2d,
        },

        {
          name: ModelNodes.ConstantPad1d,
          node: ConstantPad1d,
        },

        {
          name: ModelNodes.ConstantPad2d,
          node: ConstantPad2d,
        },

        {
          name: ModelNodes.ConstantPad3d,
          node: ConstantPad3d,
        },

      ],
    },

    {
      category: ModelCategories.Dropout,
      nodes: [
        {
          name: ModelNodes.Dropout,
          node: Dropout,
        },
        {
          name: ModelNodes.Dropout2d,
          node: Dropout2d,
        },
        {
          name: ModelNodes.Dropout3d,
          node: Dropout3d,
        },
        {
          name: ModelNodes.AlphaDropout,
          node: AlphaDropout,
        },
      ],
    },
    {
      category: ModelCategories.Activation,
      nodes: [

        {
          name: ModelNodes.ELU,
          node: ELU,
        },

        {
          name: ModelNodes.Hardshrink,
          node: Hardshrink,
        },

        {
          name: ModelNodes.Hardsigmoid,
          node: Hardsigmoid,
        },

        {
          name: ModelNodes.Hardtanh,
          node: Hardtanh,
        },

        {
          name: ModelNodes.Hardswish,
          node: Hardswish,
        },

        {
          name: ModelNodes.LeakyReLU,
          node: LeakyReLU,
        },

        {
          name: ModelNodes.MultiheadAttention,
          node: MultiheadAttention,
        },

        {
          name: ModelNodes.PReLU,
          node: PReLU,
        },

        {
          name: ModelNodes.Relu,
          node: ReLU,
        },

        {
          name: ModelNodes.ReLU6,
          node: ReLU6,
        },

        {
          name: ModelNodes.RReLU,
          node: RReLU,
        },

        {
          name: ModelNodes.SELU,
          node: SELU,
        },

        {
          name: ModelNodes.CELU,
          node: CELU,
        },

        {
          name: ModelNodes.GELU,
          node: GELU,
        },

        {
          name: ModelNodes.Sigmoid,
          node: Sigmoid,
        },

        {
          name: ModelNodes.SiLU,
          node: SiLU,
        },

        {
          name: ModelNodes.Softplus,
          node: Softplus,
        },

        {
          name: ModelNodes.Softshrink,
          node: Softshrink,
        },

        {
          name: ModelNodes.Softsign,
          node: Softsign,
        },

        {
          name: ModelNodes.Tanh,
          node: Tanh,
        },

        {
          name: ModelNodes.LogSigmoid,
          node: LogSigmoid,
        },

        {
          name: ModelNodes.Tanhshrink,
          node: Tanhshrink,
        },

        {
          name: ModelNodes.Threshold,
          node: Threshold,
        },

      ],
    },
    {
      category: ModelCategories.Operations,
      nodes: [
        {
          name: ModelNodes.Concat,
          node: Concat,
        },
        {
          name: ModelNodes.Flatten,
          node: Flatten,
        },
      ],
    },
    {
      category: ModelCategories.Linear,
      nodes: [
        {
          name: ModelNodes.Linear,
          node: Linear,
        },
        {
          name: ModelNodes.Bilinear,
          node: Bilinear,
        },
      ],
    },
    {
      category: ModelCategories.NonLinearActivation,
      nodes: [
        {
          name: ModelNodes.Softmin,
          node: Softmin,
        },
        {
          name: ModelNodes.Softmax,
          node: Softmax,
        },
        {
          name: ModelNodes.LogSoftmax,
          node: LogSoftmax,
        },

        {
          name: ModelNodes.Softmax2d,
          node: Softmax2d,
        },

        {
          name: ModelNodes.AdaptiveLogSoftmaxWithLoss,
          node: AdaptiveLogSoftmaxWithLoss,
        },

      ],
    },

    {
      category: ModelCategories.Normalization,
      nodes: [

        {
          name: ModelNodes.BatchNorm1d,
          node: BatchNorm1d,
        },

        {
          name: ModelNodes.BatchNorm2d,
          node: BatchNorm2d,
        },

        {
          name: ModelNodes.BatchNorm3d,
          node: BatchNorm3d,
        },

        {
          name: ModelNodes.GroupNorm,
          node: GroupNorm,
        },

        {
          name: ModelNodes.SyncBatchNorm,
          node: SyncBatchNorm,
        },

        {
          name: ModelNodes.InstanceNorm1d,
          node: InstanceNorm1d,
        },

        {
          name: ModelNodes.InstanceNorm2d,
          node: InstanceNorm2d,
        },

        {
          name: ModelNodes.InstanceNorm3d,
          node: InstanceNorm3d,
        },

        {
          name: ModelNodes.LocalResponseNorm,
          node: LocalResponseNorm,
        },
      ],
    },
    {
      category: ModelCategories.RecurrentLayers,
      nodes: [

        {
          name: ModelNodes.RNNBase,
          node: RNNBase,
        },

        {
          name: ModelNodes.RNN,
          node: RNN,
        },

        {
          name: ModelNodes.LSTM,
          node: LSTM,
        },

        {
          name: ModelNodes.GRU,
          node: GRU,
        },

        {
          name: ModelNodes.RNNCell,
          node: RNNCell,
        },

        {
          name: ModelNodes.LSTMCell,
          node: LSTMCell,
        },

        {
          name: ModelNodes.GRUCell,
          node: GRUCell,
        },

      ],
    },
    {
      category: ModelCategories.Transformer,
      nodes: [
        {
          name: ModelNodes.Transformer,
          node: Transformer,
        },
        {
          name: ModelNodes.TransformerEncoderLayer,
          node: TransformerEncoderLayer,
        },

        {
          name: ModelNodes.TransformerDecoderLayer,
          node: TransformerDecoderLayer,
        },

      ],
    },

    {
      category: ModelCategories.DistanceFunctions,
      nodes: [
        {
          name: ModelNodes.CosineSimilarity,
          node: CosineSimilarity,
        },

        {
          name: ModelNodes.PairwiseDistance,
          node: PairwiseDistance,
        },
      ],
    },

    {
      category: ModelCategories.SparseLayers,
      nodes: [
        {
          name: ModelNodes.Embedding,
          node: Embedding,
        },

        {
          name: ModelNodes.EmbeddingBag,
          node: EmbeddingBag,
        },
      ],
    },
  ];

  customNodeType = ModelCustom;
  customNodeName = ModelNodes.ModelCustom;

  public registerNodes(editor: Editor) {
    super.registerNodes(editor);
    editor.registerNodeType(ModelNodes.InModel, InModel);
    editor.registerNodeType(ModelNodes.OutModel, OutModel);
  }
}
