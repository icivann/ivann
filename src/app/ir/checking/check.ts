import Graph from '@/app/ir/Graph';
import IrError from '@/app/ir/checking/irError';
import { findBadConvChannels } from '@/app/ir/checking/findBadConvChannels';
import { findDanglingInterfaces } from './findDanglingInterfaces';

export function check(graph: Graph): IrError[] {
  const errorsSoFar = Array<IrError>();
  errorsSoFar.push(...findDanglingInterfaces(graph));
  errorsSoFar.push(...findBadConvChannels(graph));
  return errorsSoFar;
}
