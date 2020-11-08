class Concat {
  constructor() {}
  static build(options: Map<string, any>): Concat {
    return new Concat();
  }

  public callCode(params: string[], name: string) {
    return `torch.cat(${params.join(', ')})`;
  }
}

export default Concat;
