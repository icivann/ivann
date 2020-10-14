/* eslint-disable */

class CarExample {
  constructor(
    public readonly wheels: number,
    public readonly topSpeed: number,
    public readonly name: string,
    public readonly isComfortable: boolean,
  ) {

  }

  public copyWith(modifyObject: { [P in keyof CarExample]?: CarExample[P] }): CarExample {
    return Object.assign(Object.create(CarExample.prototype), {...this, ...modifyObject});
  }
}
