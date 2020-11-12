class OutModel {
  constructor(
    public readonly name: string,
  ) {}

  static build(options: Map<string, any>): OutModel {
    return new OutModel(
      options.get('name'),
    );
  }
}

export default OutModel;
