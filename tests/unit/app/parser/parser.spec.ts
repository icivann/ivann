import parse from '@/app/parser/parser';
import ParsedFunction from '@/app/parser/ParsedFunction';

describe('parser', () => {
  it('parses a single function with args correctly', () => {
    const text = 'def func(arg1, arg2):\n'
      + '  line1\n'
      + '  line2\n';

    const expected = [
      new ParsedFunction(
        'func',
        '  line1\n  line2\n',
        ['arg1', 'arg2'],
      ),
    ];

    const res = parse(text);
    expect(res).toEqual(expected);
  });

  it('parses a function with no final line', () => {
    const text = 'def func(arg1, arg2):\n'
      + '  line1\n'
      + '  line2';

    const expected = [
      new ParsedFunction(
        'func',
        '  line1\n  line2\n',
        ['arg1', 'arg2'],
      ),
    ];

    const res = parse(text);
    expect(res).toEqual(expected);
  });

  it('parses a single function without args correctly', () => {
    const text = 'def func():\n'
      + '  line1\n'
      + '  line2\n';

    const expected = [
      new ParsedFunction(
        'func',
        '  line1\n  line2\n',
        [],
      ),
    ];

    const res = parse(text);
    expect(res).toEqual(expected);
  });

  it('parses a tab indent correctly', () => {
    const text = 'def func():\n'
      + '\tline1\n'
      + '\tline2\n';

    const expected = [
      new ParsedFunction(
        'func',
        '\tline1\n\tline2\n',
        [],
      ),
    ];

    const res = parse(text);
    expect(res).toEqual(expected);
  });

  it('parses two functions correctly', () => {
    const text = 'def func():\n'
      + '\tline1\n'
      + '\tline2\n'
      + 'line3\n'
      + 'def func2(arg):\n'
      + '\tline4\n'
      + '\tline5\n'
      + 'line6\n';

    const expected = [
      new ParsedFunction(
        'func',
        '\tline1\n\tline2\n',
        [],
      ),
      new ParsedFunction(
        'func2',
        '\tline4\n\tline5\n',
        ['arg'],
      ),
    ];

    const res = parse(text);
    expect(res).toEqual(expected);
  });

  it('parses a function with multiple levels of indentation', () => {
    const text = 'def func():\n'
      + '  line1\n'
      + '    line2\n'
      + '  line3\n';

    const expected = [
      new ParsedFunction(
        'func',
        '  line1\n    line2\n  line3\n',
        [],
      ),
    ];

    const res = parse(text);
    expect(res).toEqual(expected);
  });
  it('fails to parse a non file with no functions', () => {
    const text = 'line1\n'
      + '  line2\n'
      + '  line3\n';

    expect(parse(text)).toBeInstanceOf(Error);
  });

  it('fails to parse a file with a badly indented function', () => {
    const text = 'line1\n'
      + '  def func(arg1):\n'
      + '    line3\n';

    expect(parse(text)).toBeInstanceOf(Error);
  });

  it('fails to parse a badly formatted function signature 1', () => {
    const text = 'def funcarg1, arg2):\n'
      + '  line2\n'
      + '  line3\n';

    expect(parse(text)).toBeInstanceOf(Error);
  });

  it('fails to parse a badly formatted function signature 2', () => {
    const text = 'def func(arg1, arg2:\n'
      + '  line2\n'
      + '  line3\n';

    expect(parse(text)).toBeInstanceOf(Error);
  });

  it('fails to parse a badly formatted function signature 3', () => {
    const text = 'def func(arg1, arg2)\n'
      + '  line2\n'
      + '  line3\n';

    expect(parse(text)).toBeInstanceOf(Error);
  });

  it('fails to parse a badly formatted function signature 4', () => {
    const text = 'deffunc(arg1, arg2)\n'
      + '  line2\n'
      + '  line3\n';

    expect(parse(text)).toBeInstanceOf(Error);
  });
});
