export enum Padding {Valid, Same}

export type Initializer = CustomInitializer | BuiltinInitializer
enum BuiltinInitializer {}
export class CustomInitializer {
}

export type Regularizer = BuiltinRegularizer
enum BuiltinRegularizer {}

export type ActivationF = BuiltinActivationF
enum BuiltinActivationF { Relu }
