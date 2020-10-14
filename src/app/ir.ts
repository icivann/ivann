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
    return Object.assign(Object.create(CarExample.prototype), { ...this, ...modifyObject });
  }
}

const c = new CarExample(3, 3, 'asd', false);

const c2 = c.copyWith({ wheels: 4, topSpeed: 100 });

type MLNode = ModelNode | TrainingNode

class GraphNode implements IvannNode {
  constructor(
    public readonly wheels: number,
  ) {
  }
  public readonly uniqueId = randomUuid()
}

class ModelNode {}
class TrainingNode {}
interface IvannNode {
  uniqueId: string
}
