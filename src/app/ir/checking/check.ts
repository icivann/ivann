import Graph from '@/app/ir/Graph';
import IrError from '@/app/ir/checking/irError';
import { findDanglingInterfaces } from './findDanglingInterfaces';

export function check(graph: Graph): IrError[] {
  const errorsSoFar = Array<IrError>();
  errorsSoFar.push(...findDanglingInterfaces(graph));

  return errorsSoFar;
}
