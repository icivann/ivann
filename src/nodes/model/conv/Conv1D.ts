import Conv, { ConvOptions } from '@/nodes/model/conv/Conv';
import { Nodes } from '@/nodes/model/Types';

export default class Conv1D extends Conv {
  type = Nodes.Conv1D;
  name = Nodes.Conv1D;

  protected addKernelStride(): void {
    // TODO: Keras+Pytorch allow shortcut for specifying single int for all dimensions
    this.addOption(ConvOptions.KernelSize, 'IntOption', 1, undefined, {
      min: 1,
    });
    this.addOption(ConvOptions.Stride, 'IntOption', 1, undefined, {
      min: 1,
    });
  }
}
