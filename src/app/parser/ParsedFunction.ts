class ParsedFunction {
  constructor(
    public readonly name: string,
    public readonly body: string,
    public readonly args: string[],
    public readonly filename?: string,
  ) {
  }

  public signature(): string {
    return `def ${this.name}(${this.args.join(', ')})`;
  }

  public toString(): string {
    return `${this.signature()}:\n${this.body}`;
  }

  public equals(other: ParsedFunction): boolean {
    if (this.args.length !== other.args.length) return false;

    const sortedArgs = this.args.slice().sort();
    const otherSortedArgs = other.args.slice().sort();
    for (let i = 0; i < this.args.length; i += 1) {
      if (sortedArgs[i] !== otherSortedArgs[i]) return false;
    }

    return this.name === other.name && this.body === other.body && this.filename === other.filename;
  }

  public containsReturn(): boolean {
    return this.body.includes('return');
  }
}

export default ParsedFunction;
