class Model {
  constructor(
    public readonly name: string,
  ) {
  }

  static build(options: Map<string, any>): Model {
    return new Model(
      options.get('name'),
    );
  }
}

export default Model;
