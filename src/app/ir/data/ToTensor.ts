import { nodeName } from '@/app/ir/irCommon';

class ToTensor {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): ToTensor {
    return new ToTensor(
      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'transforms.ToTensor()';
  }
}

export default ToTensor;
