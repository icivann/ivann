export enum Padding {Valid, Same}

export enum PaddingMode {
  zeros = 'zeros', reflect = 'reflect', replicate = 'replicate', circular = 'circular'
}

export const nodeName = 'name';

export type Initializer = CustomInitializer | BuiltinInitializer

export enum BuiltinInitializer {
  Zeroes, Ones, Xavier
}

export class CustomInitializer {
}

export type Regularizer = BuiltinRegularizer

export enum BuiltinRegularizer {
  None
}

export enum Reduction{
  None ='none', Mean = 'mean', Sum = 'sum'
}
export function getReduction(str: string): Reduction {
  return Reduction[str as keyof typeof Reduction];
}
export enum Mode{
  None ='none', Mean = 'mean', Sum = 'sum'
}
export function getMode(str: string): Mode {
  return Mode[str as keyof typeof Mode];
}

export type ActivationF = BuiltinActivationF

export enum Activation {
  Relu='Relu', Gelu='Gelu'
}

export enum BuiltinActivationF { None, Relu, Tanh, Sigmoid, Linear }

export function getRegularizer(str: string): Regularizer {
  return BuiltinRegularizer[str as keyof typeof BuiltinRegularizer];
}

export function getInitializer(str: string): Initializer {
  return BuiltinInitializer[str as keyof typeof BuiltinInitializer];
}

export function getActivation(str: string): Activation {
  return Activation[str as keyof typeof Activation];
}

export function getPadding(str: string): Padding {
  return Padding[str as keyof typeof Padding];
}
