import GraphNode from '@/app/ir/GraphNode';
import { Connection } from '@/app/ir/Graph';
import { Severity } from '@/app/ir/checking/severity';

export default class IrError {
  constructor(
        public readonly offenders: (GraphNode | Connection)[],
        public readonly severity: Severity,
        public readonly message: string,
  ) {
  }

  get formattedMessage(): string {
    return `- ${this.severity}: ${this.message}`;
  }
}
