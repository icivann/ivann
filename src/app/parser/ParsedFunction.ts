class ParsedFunction {
  constructor(
    public readonly name: string,
    public readonly body: string,
    public readonly args: string[],
  ) {
  }
}

export default ParsedFunction;
