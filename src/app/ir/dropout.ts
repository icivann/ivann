import { DropoutOptions } from '@/nodes/model/regularization/Dropout';

export default class Dropout {
  constructor(
    public readonly probability: number,
  ) {}

  static build(options: Map<string, any>): Dropout {
    return new Dropout(
      options.get(DropoutOptions.Probability),
    );
  }
}
