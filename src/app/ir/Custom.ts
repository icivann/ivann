class Custom {
  constructor(
    public readonly code: string,
  ) {
  }

  static build(options: Map<string, any>): Custom {
    // TODO CORE-58 Change InlineCode Option to use state.parsedFunction
    return new Custom(
      // options.get(CustomOptions.InlineCode).text,
      '',
    );
  }
}

export default Custom;
