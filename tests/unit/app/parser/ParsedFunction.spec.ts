import ParsedFunction from '@/app/parser/ParsedFunction';

describe('parsedFunction', () => {
  it('stringifies a function with args correctly', () => {
    const parsedFunction = new ParsedFunction(
      'func',
      '  line1\n  line2\n',
      ['arg1', 'arg2'],
    );

    const expected = 'def func(arg1, arg2):\n'
      + '  line1\n'
      + '  line2\n';

    expect(parsedFunction.toString()).toBe(expected);
  });

  it('stringifies a function without args correctly', () => {
    const parsedFunction = new ParsedFunction(
      'func',
      '  line1\n  line2\n',
      [],
    );
    const expected = 'def func():\n'
      + '  line1\n'
      + '  line2\n';

    expect(parsedFunction.toString()).toBe(expected);
  });
});
