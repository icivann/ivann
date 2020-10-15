export class UUID {
  constructor(public readonly id: string) {
  }
}

export function randomUuid(): UUID {
  const s = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const
      v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
  return new UUID(s);
}

export type Option<T> = null | T

export function valuesOf(obj: { [s: string]: any }): string[] {
  return Object.values(obj).filter((t) => typeof t === 'string');
}
