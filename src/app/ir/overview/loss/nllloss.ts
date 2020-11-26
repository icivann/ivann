import { nodeName } from '@/app/ir/irCommon';

export default class NLLLoss {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): NLLLoss {
    return new NLLLoss(
      options.get(nodeName),
    );
  }

  public initCode(): string {
    return 'NLLLoss()';
  }
}
