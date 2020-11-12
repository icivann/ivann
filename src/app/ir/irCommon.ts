export enum Padding {Valid, Same}

export enum PaddingMode {
  zeros = 'zeros', reflect = 'reflect', replicate = 'replicate', circular = 'circular'
}

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

export type ActivationF = BuiltinActivationF

export enum BuiltinActivationF { None, Relu, Tanh, Sigmoid, Linear }

export function getRegularizer(str: string): Regularizer {
  return BuiltinRegularizer[str as keyof typeof BuiltinRegularizer];
}

export function getInitializer(str: string): Initializer {
  return BuiltinInitializer[str as keyof typeof BuiltinInitializer];
}

export function getBuiltinActivationFunction(str: string): BuiltinActivationF {
  return BuiltinActivationF[str as keyof typeof BuiltinActivationF];
}

export function getPadding(str: string): Padding {
  return Padding[str as keyof typeof Padding];
}
