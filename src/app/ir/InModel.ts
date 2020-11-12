class InModel {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): InModel {
    return new InModel(
      options.get('name'),
    );
  }
}

export default InModel;
