import { CustomOptions } from '@/nodes/model/custom/Custom';

class Custom {
  public readonly name = 'custom'
  constructor(
    public readonly code: string,
  ) {
  }

  static build(options: Map<string, any>): Custom {
    return new Custom(
      options.get(CustomOptions.InlineCode).text,
    );
  }
}

export default Custom;
