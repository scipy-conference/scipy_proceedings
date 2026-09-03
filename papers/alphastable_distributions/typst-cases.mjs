// Project-local MyST compatibility transform. MyST currently drops LaTeX
// `cases` environments during Typst export, translates the TeX shorthand `\|`
// to a single Typst bar, and loses the header semantics of table sections after
// an empty full-width `\multicolumn` separator. Repair those constructs in the
// document tree before export.
// This file intentionally has no npm dependencies so uvx builds remain
// reproducible from a clean checkout.
import { readFileSync } from 'node:fs';

const CASES_BEGIN = String.raw`\begin{cases}`;
const CASES_END = String.raw`\end{cases}`;
const DOUBLE_BAR = String.raw`\|`;
const NEGATIVE_THIN_SPACE = String.raw`\!`;
// Keep a terminating space because the shorthand may be immediately followed
// by a letter (for example `\left\|c`); TeX ignores it after the command name.
const DOUBLE_BAR_NAMED = String.raw`\Vert `;

const COMMANDS = {
  alpha: 'alpha',
  beta: 'beta',
  delta: 'delta',
  gamma: 'gamma',
  kappa: 'kappa',
  pi: 'pi',
  Sigma: 'Sigma',
  Gamma: 'Gamma',
  Vert: 'bar.v.double',
  infty: 'infinity',
  neq: '!=',
  ne: '!=',
  geq: '>=',
  leq: '<=',
  exp: 'exp',
  cos: 'cos',
  tan: 'tan',
  log: 'log',
  ln: 'ln',
  ii: 'ii',
  Pr: 'Pr',
};

class LatexSubsetConverter {
  constructor(value) {
    this.value = value;
    this.index = 0;
  }

  convert(stop = undefined) {
    let output = '';

    const append = (token) => {
      if (!token) return;
      // TeX treats adjacent letters as separate math atoms, whereas Typst
      // parses them as one identifier (for example, iuX or ii beta).
      if (/[A-Za-z0-9)\]"]$/.test(output) && /^[A-Za-z]/.test(token))
        output += ' ';
      output += token;
    };

    while (this.index < this.value.length) {
      const char = this.value[this.index];
      if (char === stop) {
        this.index += 1;
        break;
      }
      if (/\s/.test(char)) {
        this.index += 1;
        if (output && !output.endsWith(' ')) output += ' ';
        continue;
      }
      if (char === '\\') {
        append(this.convertCommand());
        continue;
      }
      if (char === '{') {
        this.index += 1;
        append(`(${this.convert('}').trim()})`);
        continue;
      }
      if (char === '^' || char === '_') {
        this.index += 1;
        append(`${char}(${this.convertArgument()})`);
        continue;
      }
      if (char === ',') {
        append('comma');
        this.index += 1;
        continue;
      }
      append(char);
      this.index += 1;
    }

    return output.replace(/\s+/g, ' ').trim();
  }

  commandName() {
    this.index += 1;
    const match = this.value.slice(this.index).match(/^[A-Za-z]+/);
    if (match) {
      this.index += match[0].length;
      return match[0];
    }
    const name = this.value[this.index] ?? '';
    this.index += 1;
    return name;
  }

  skipWhitespace() {
    while (/\s/.test(this.value[this.index] ?? '')) this.index += 1;
  }

  rawArgument() {
    this.skipWhitespace();
    if (this.value[this.index] !== '{') {
      const value = this.value[this.index] ?? '';
      this.index += 1;
      return value;
    }

    this.index += 1;
    const start = this.index;
    let depth = 1;
    while (this.index < this.value.length && depth > 0) {
      if (this.value[this.index] === '{' && this.value[this.index - 1] !== '\\')
        depth += 1;
      if (this.value[this.index] === '}' && this.value[this.index - 1] !== '\\')
        depth -= 1;
      this.index += 1;
    }
    return this.value.slice(start, this.index - 1);
  }

  convertArgument() {
    this.skipWhitespace();
    if (this.value[this.index] === '{') {
      return new LatexSubsetConverter(this.rawArgument()).convert();
    }
    if (this.value[this.index] === '\\') return this.convertCommand();

    const value = this.value[this.index] ?? '';
    this.index += 1;
    return value;
  }

  convertDelimiter() {
    this.skipWhitespace();
    if (this.value[this.index] === '\\') {
      const name = this.commandName();
      return { lbrace: '{', rbrace: '}', vert: '|', Vert: '||' }[name] ?? name;
    }
    const delimiter = this.value[this.index] ?? '';
    this.index += 1;
    return delimiter === '.' ? '' : delimiter;
  }

  convertCommand() {
    const name = this.commandName();
    if (COMMANDS[name]) return COMMANDS[name];
    if (['displaystyle', 'textstyle'].includes(name)) return '';
    if ([',', ';', ':', '!'].includes(name)) return ' ';
    if (name === 'quad') return 'quad';
    if (name === 'qquad') return 'quad quad';
    if (
      ['left', 'right', 'big', 'bigl', 'bigr', 'Big', 'Bigl', 'Bigr'].includes(
        name,
      )
    ) {
      return this.convertDelimiter();
    }
    if (['frac', 'dfrac', 'tfrac'].includes(name)) {
      return `frac(${this.convertArgument()}, ${this.convertArgument()})`;
    }
    if (name === 'sqrt') return `sqrt(${this.convertArgument()})`;
    if (name === 'overset') {
      const above = this.convertArgument();
      const base = this.convertArgument();
      return `overset(${base}, ${above})`;
    }
    if (name === 'operatorname')
      return `op(${JSON.stringify(this.rawArgument().trim())})`;
    if (name === 'text') return JSON.stringify(this.rawArgument().trim());
    if (name === 'mathrm') return `upright(${this.convertArgument()})`;
    if (name === 'mathbb') {
      const value = this.rawArgument().trim();
      return /^[A-Z]$/.test(value) ? value.repeat(2) : value;
    }
    if (name === '&') return 'amp';
    if (['{', '}', '|'].includes(name)) return name;
    return name;
  }
}

function splitTopLevel(value, separator) {
  const parts = [];
  let start = 0;
  let braceDepth = 0;

  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (char === '{' && value[index - 1] !== '\\') braceDepth += 1;
    if (char === '}' && value[index - 1] !== '\\') braceDepth -= 1;

    if (braceDepth === 0 && value.startsWith(separator, index)) {
      parts.push(value.slice(start, index));
      index += separator.length - 1;
      start = index + 1;
    }
  }

  parts.push(value.slice(start));
  return parts;
}

function convertLatex(value) {
  return new LatexSubsetConverter(value).convert();
}

function convertCasesBody(body) {
  const rows = splitTopLevel(body, String.raw`\\`)
    .map((row) => row.trim())
    .filter(Boolean)
    .map((row) => {
      const cells = splitTopLevel(row, '&').map((cell) => convertLatex(cell));
      return cells.length > 1
        ? `${cells[0]} & ${cells.slice(1).join(' ')}`
        : cells[0];
    });

  return `cases(${rows.map((row) => `${row},`).join(' ')})`;
}

function convertMathWithCases(value) {
  let remaining = value;
  let typst = '';

  while (remaining.includes(CASES_BEGIN)) {
    const begin = remaining.indexOf(CASES_BEGIN);
    const end = remaining.indexOf(CASES_END, begin + CASES_BEGIN.length);
    if (end < 0) return undefined;

    typst += `${convertLatex(remaining.slice(0, begin))} `;
    typst += convertCasesBody(remaining.slice(begin + CASES_BEGIN.length, end));
    remaining = remaining.slice(end + CASES_END.length);
  }

  typst += ` ${convertLatex(remaining)}`;
  return typst.trim();
}

function hasVisibleContent(node) {
  if (!node || typeof node !== 'object') return false;
  if (node.type === 'thematicBreak') return true;
  if (node.type === 'raw' && typeof node.typst === 'string') return true;
  if (typeof node.value === 'string' && node.value.trim()) return true;
  return Array.isArray(node.children) && node.children.some(hasVisibleContent);
}

function typstTableRule() {
  return {
    type: 'raw',
    typst: '#line(length: 100%, stroke: gray)',
  };
}

function boldTableRow(row) {
  row.children?.forEach((cell) => {
    if (!Array.isArray(cell.children) || cell.children.length === 0) return;
    if (cell.children.length === 1 && cell.children[0]?.type === 'strong')
      return;
    cell.children = [{ type: 'strong', children: cell.children }];
  });
}

function typstTableRuleRow(columnCount) {
  return {
    type: 'tableRow',
    children: [
      {
        type: 'tableCell',
        colspan: columnCount,
        align: 'center',
        children: [typstTableRule()],
      },
    ],
  };
}

function restoreTableSectionHeaders(tree, utils) {
  utils.selectAll('table', tree).forEach((table) => {
    if (!Array.isArray(table.children)) return;

    let firstHeaderPadded = false;

    for (let index = 1; index < table.children.length; index += 1) {
      const separator = table.children[index - 1];
      const header = table.children[index];
      if (
        separator?.type !== 'tableRow' ||
        header?.type !== 'tableRow' ||
        separator.children?.length !== 1 ||
        separator.children[0]?.type !== 'tableCell' ||
        !(separator.children[0].colspan > 1) ||
        hasVisibleContent(separator.children[0]) ||
        !Array.isArray(header.children)
      )
        continue;

      // LaTeX uses the empty spanning row as a visual section boundary. Keep
      // the explicit line Typst-only: web themes apply large block margins to
      // thematic breaks inside table cells, producing oversized empty rows.
      // HTML already draws the table row borders needed at this boundary.
      separator.children[0].children = [typstTableRule()];

      // MyST otherwise treats the next section header as ordinary body data.
      boldTableRow(header);

      // The first title must receive the same explicit bold treatment as the
      // restored section title. Matching spacer rows are added after scanning
      // the table so they do not interfere with section-header detection.
      if (!firstHeaderPadded) {
        const firstHeader = table.children[0];
        if (firstHeader?.type === 'tableRow') {
          boldTableRow(firstHeader);
          firstHeaderPadded = true;
        }
      }

      const columnCount = header.children.reduce(
        (count, cell) => count + (cell.colspan ?? 1),
        0,
      );
      table.children.splice(index + 1, 0, typstTableRuleRow(columnCount));
      index += 1;
    }

    if (firstHeaderPadded) {
      const firstHeader = table.children[0];
      const columnCount = firstHeader.children.reduce(
        (count, cell) => count + (cell.colspan ?? 1),
        0,
      );
      // Mirror the second section exactly: spacer/rule, title, spacer/rule.
      table.children.splice(0, 0, typstTableRuleRow(columnCount));
      table.children.splice(2, 0, typstTableRuleRow(columnCount));
    }
  });
}

function latexSectionNumber(source, offset) {
  let section = 0;
  let appendix = false;
  let appendixSection = 0;
  const prefix = source.slice(0, offset);

  for (const line of prefix.split('\n')) {
    const code = line.replace(/(^|[^\\])%.*/, '$1');
    if (/\\appendix\b/.test(code)) {
      appendix = true;
      appendixSection = 0;
    }
    if (!/\\section\s*\{/.test(code)) continue;
    if (appendix) appendixSection += 1;
    else section += 1;
  }

  return appendix
    ? String.fromCharCode('A'.charCodeAt(0) + appendixSection - 1)
    : String(section);
}

function latexProofRules(source) {
  const rules = new Map();
  if (typeof source !== 'string') return rules;

  // \newtheorem{definition}{Definition}[section]
  const pattern =
    /\\newtheorem\s*\{([^}]+)\}(?:\s*\[([^\]]+)\])?\s*\{([^}]+)\}(?:\s*\[([^\]]+)\])?/g;
  for (const match of source.matchAll(pattern)) {
    rules.set(match[1].toLowerCase(), {
      shared: match[2]?.toLowerCase(),
      title: match[3],
      within: match[4]?.toLowerCase(),
    });
  }
  return rules;
}

function restoreNonFloatingProofEnvironments(tree, utils, source) {
  const rules = latexProofRules(source);
  const counters = new Map();
  const numbersByLabel = new Map();
  utils.selectAll('proof', tree).forEach((node) => {
    if (!Array.isArray(node.children)) return;

    const kind = node.kind?.toLowerCase();
    const rule = rules.get(kind);
    if (!rule && !['definition', 'theorem'].includes(kind)) return;

    const offset = node.position?.start?.offset ?? 0;
    const within = rule?.within;
    const prefix =
      within === 'section' ? latexSectionNumber(source, offset) : '';
    const counterName = rule?.shared ?? kind;
    const counterKey = `${counterName}:${prefix}`;
    const next = (counters.get(counterKey) ?? 0) + 1;
    counters.set(counterKey, next);

    const number = prefix ? `${prefix}.${next}` : String(next);
    const supplement =
      rule?.title ?? kind[0].toUpperCase() + kind.slice(1).toLowerCase();
    const title = `${supplement} ${number}.`;
    node.enumerator = String(number);
    if (node.identifier)
      numbersByLabel.set(node.identifier.toLowerCase(), String(number));
    const label = node.identifier
      ? [
          {
            type: 'raw',
            typst: `#metadata(none) <${node.identifier}>\n`,
          },
        ]
      : [];

    // MyST's Typst proof helper hard-codes `float: true`, unlike LaTeX theorem
    // and definition environments. Use an equivalent non-floating admonition
    // so each environment remains at its source position. Keep an explicit
    // label for cross-references because the proof node normally emits it.
    node.type = 'admonition';
    node.kind = 'important';
    node.children = [
      {
        type: 'admonitionTitle',
        children: [{ type: 'text', value: title }],
      },
      ...label,
      ...node.children,
    ];
  });

  utils.selectAll('crossReference', tree).forEach((node) => {
    const number = numbersByLabel.get(node.identifier?.toLowerCase());
    if (!number) return;
    node.enumerator = number;
    node.children = [{ type: 'text', value: number }];
  });
}

function latexEquationNumbers(source) {
  const environments = [];
  const numbersByLabel = new Map();
  if (typeof source !== 'string') return { environments, numbersByLabel };

  let counter = 0;
  const pattern = /\\begin\{(equation\*?|align\*?)\}([\s\S]*?)\\end\{\1\}/g;
  for (const match of source.matchAll(pattern)) {
    const name = match[1];
    const body = match[2];
    const entry = {
      start: match.index,
      end: match.index + match[0].length,
      enumerated: !name.endsWith('*'),
      number: undefined,
    };

    if (!entry.enumerated) {
      environments.push(entry);
      continue;
    }

    const rows = name === 'align' ? body.split(/\\\\(?=\s*(?:&|$))/m) : [body];
    for (const row of rows) {
      if (!row.trim() || /\\(?:nonumber|notag)\b/.test(row)) continue;
      counter += 1;
      entry.number ??= String(counter);
      for (const label of commandArguments(row, 'label'))
        numbersByLabel.set(label.toLowerCase(), String(counter));
    }
    environments.push(entry);
  }

  return { environments, numbersByLabel };
}

function restoreLatexEquationNumbering(tree, utils, source) {
  if (typeof source !== 'string') return;
  const { environments, numbersByLabel } = latexEquationNumbers(source);

  utils.selectAll('math', tree).forEach((node) => {
    const offset = node.position?.start?.offset;
    if (typeof offset !== 'number') return;
    const environment = environments.find(
      (candidate) => offset >= candidate.start && offset < candidate.end,
    );

    // TeX only numbers equation/align (without a star). MyST otherwise assigns
    // counters to every display written with \[...\], shifting later refs.
    const mathNode = { ...node };
    delete mathNode.key;
    for (const key of Object.keys(node)) delete node[key];
    node.type = 'div';

    if (!environment || !environment.enumerated) {
      // The SciPy Typst template numbers every block equation globally. Keep
      // LaTeX's \[...\] and starred environments in a local unnumbered scope.
      mathNode.enumerated = false;
      delete mathNode.enumerator;
      node.children = [
        {
          type: 'raw',
          typst: '#[\n#set math.equation(numbering: none)\n',
        },
        mathNode,
        { type: 'raw', typst: '\n]\n' },
      ];
      return;
    }

    mathNode.enumerated = true;
    const labelNumber = numbersByLabel.get(
      (mathNode.label ?? mathNode.identifier)?.toLowerCase(),
    );
    const number = labelNumber ?? environment.number;
    mathNode.enumerator = number;
    node.children = [
      {
        type: 'raw',
        // Typst's counter otherwise includes the unnumbered displays above.
        typst: `#counter(math.equation).update(${Number(number) - 1})\n`,
      },
      mathNode,
    ];
  });

  utils.selectAll('crossReference', tree).forEach((node) => {
    const number = numbersByLabel.get(node.identifier?.toLowerCase());
    if (!number) return;
    node.enumerator = number;
    node.children = [{ type: 'text', value: `(${number})` }];
  });
}

function restoreLatexHeadingNumbering(tree, utils, source) {
  if (typeof source !== 'string') return;
  const headings = utils
    .selectAll('heading', tree)
    .filter((node) => typeof node.position?.start?.offset === 'number')
    .sort(
      (left, right) => left.position.start.offset - right.position.start.offset,
    );
  const counters = { section: 0, subsection: 0, subsubsection: 0 };
  const numbersByLabel = new Map();
  let appendixSection = 0;

  headings.forEach((node) => {
    const offset = node.position.start.offset;
    const command = source
      .slice(offset, offset + 40)
      .match(/^\\(section|subsection|subsubsection|paragraph)(\*)?\s*\{/);
    if (!command) return;

    const kind = command[1];
    const starred = Boolean(command[2]);
    const appendix = source.lastIndexOf('\\appendix', offset) >= 0;
    let number;

    // article.cls numbers through subsubsection by default; paragraph headings
    // and starred headings are deliberately unnumbered in the TeX source.
    if (!starred && kind !== 'paragraph') {
      if (kind === 'section') {
        counters.subsection = 0;
        counters.subsubsection = 0;
        if (appendix) {
          appendixSection += 1;
          number = String.fromCharCode('A'.charCodeAt(0) + appendixSection - 1);
        } else {
          counters.section += 1;
          number = String(counters.section);
        }
      } else if (kind === 'subsection') {
        counters.subsection += 1;
        counters.subsubsection = 0;
        const section = appendix
          ? String.fromCharCode('A'.charCodeAt(0) + appendixSection - 1)
          : String(counters.section);
        number = `${section}.${counters.subsection}`;
      } else {
        counters.subsubsection += 1;
        number = `${counters.section}.${counters.subsection}.${counters.subsubsection}`;
      }
    }

    if (number && node.identifier)
      numbersByLabel.set(node.identifier.toLowerCase(), number);

    if (!starred && kind !== 'paragraph' && !appendix) {
      node.enumerator = number;
      return;
    }

    const headingNode = { ...node };
    delete headingNode.key;
    headingNode.enumerated = false;
    delete headingNode.enumerator;
    if (number)
      headingNode.children = [
        { type: 'text', value: `${number}. ` },
        ...(headingNode.children ?? []),
      ];
    for (const key of Object.keys(node)) delete node[key];
    node.type = 'div';

    if (starred || kind === 'paragraph') {
      node.children = [
        {
          type: 'raw',
          typst: '#block(above: 1.2em, below: 0.6em)[#text(weight: "bold")[',
        },
        ...(headingNode.children ?? []),
        { type: 'raw', typst: ']]\n' },
        ...(headingNode.identifier
          ? [
              {
                type: 'raw',
                typst: `#metadata(none) <${headingNode.identifier}>\n`,
              },
            ]
          : []),
      ];
      return;
    }

    node.children = [
      { type: 'raw', typst: '#[\n#set heading(numbering: none)\n' },
      headingNode,
      ...(headingNode.identifier
        ? [
            {
              type: 'raw',
              typst: `#metadata(none) <${headingNode.identifier}>\n`,
            },
          ]
        : []),
      { type: 'raw', typst: '\n]\n' },
    ];
  });

  utils.selectAll('crossReference', tree).forEach((node) => {
    const number = numbersByLabel.get(node.identifier?.toLowerCase());
    if (!number) return;
    node.enumerator = number;
    node.children = [{ type: 'text', value: number }];
  });
}

function restoreSubfigureCaptions(tree, utils) {
  utils.selectAll('container', tree).forEach((container) => {
    if (container.kind !== 'figure' || !Array.isArray(container.children))
      return;

    const subfigures = container.children.filter(
      (child) =>
        child?.type === 'container' &&
        child.subcontainer === true &&
        Array.isArray(child.children),
    );
    if (subfigures.length < 2) return;

    const captions = subfigures.map((subfigure) =>
      subfigure.children.find((child) => child?.type === 'caption'),
    );
    if (!captions.some(hasVisibleContent)) return;

    // MyST's Typst exporter currently hard-codes `caption: []` for every
    // subfigure. Build the equivalent grid in the document tree instead, so
    // the exporter can still rewrite/copy image URLs while the original TeX
    // subcaptions and labels remain visible. The parent container continues to
    // provide the numbered Figure caption.
    const columns = subfigures.length <= 3 ? subfigures.length : 2;
    const gridChildren = [
      {
        type: 'raw',
        typst: `#grid(columns: ${columns}, gutter: 8pt,\n`,
      },
    ];

    subfigures.forEach((subfigure, index) => {
      const caption = captions[index];
      const content = subfigure.children.filter(
        (child) => !['caption', 'legend'].includes(child?.type),
      );
      const letter = String.fromCharCode('a'.charCodeAt(0) + index);

      gridChildren.push({ type: 'raw', typst: '[\n' });
      content.forEach((child) => {
        if (child?.type === 'image')
          gridChildren.push({ type: 'raw', typst: '#' });
        gridChildren.push(child);
      });
      gridChildren.push({
        type: 'raw',
        typst: `#align(center)[#set par(justify: false)\n#text(size: 7pt)[(${letter}) `,
      });
      if (caption?.children) gridChildren.push(...caption.children);
      gridChildren.push({ type: 'raw', typst: ']\n]\n' });
      if (subfigure.identifier) {
        gridChildren.push({
          type: 'raw',
          typst: `#metadata(none) <${subfigure.identifier}>\n`,
          // Raw Typst ignores children, but retaining a lightweight copy of
          // the original target lets MyST resolve and number subfigure xrefs.
          children: [{ ...subfigure, children: [] }],
        });
      }
      gridChildren.push({ type: 'raw', typst: '],\n' });
    });

    gridChildren.push({ type: 'raw', typst: ')\n' });
    container.children = [
      { type: 'div', children: gridChildren },
      ...container.children.filter((child) => !subfigures.includes(child)),
    ];
  });
}

function commandArguments(value, command) {
  const argumentsFound = [];
  const marker = `\\${command}`;
  let searchFrom = 0;

  while (searchFrom < value.length) {
    const commandStart = value.indexOf(marker, searchFrom);
    if (commandStart < 0) break;
    let index = commandStart + marker.length;
    while (/\s/.test(value[index] ?? '')) index += 1;
    if (value[index] !== '{') {
      searchFrom = index;
      continue;
    }

    const argumentStart = ++index;
    let depth = 1;
    while (index < value.length && depth > 0) {
      if (value[index] === '{' && value[index - 1] !== '\\') depth += 1;
      if (value[index] === '}' && value[index - 1] !== '\\') depth -= 1;
      index += 1;
    }
    if (depth === 0)
      argumentsFound.push(value.slice(argumentStart, index - 1).trim());
    searchFrom = index;
  }

  return argumentsFound;
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function captionMathHtml(value) {
  const alphaEquality = value.match(/^\s*\\alpha\s*=\s*([0-9.]+)\s*$/);
  if (alphaEquality) {
    return `<span class="katex"><span class="katex-html" aria-hidden="true"><span class="base"><span class="mord mathnormal">α</span><span class="mspace" style="margin-right:0.2778em"></span><span class="mrel">=</span><span class="mspace" style="margin-right:0.2778em"></span><span class="mord">${escapeHtml(alphaEquality[1])}</span></span></span></span>`;
  }

  // This fallback remains readable if another grouped-table caption uses a
  // math expression outside the small subset handled above.
  return `<span class="katex">${escapeHtml(value)}</span>`;
}

function captionFromLatex(value) {
  const children = [];
  let index = 0;

  while (index < value.length) {
    const mathStart = value.indexOf('$', index);
    if (mathStart < 0) {
      if (value.slice(index))
        children.push({ type: 'text', value: value.slice(index) });
      break;
    }
    if (mathStart > index)
      children.push({ type: 'text', value: value.slice(index, mathStart) });
    const mathEnd = value.indexOf('$', mathStart + 1);
    if (mathEnd < 0) {
      children.push({ type: 'text', value: value.slice(mathStart) });
      break;
    }
    const mathValue = value.slice(mathStart + 1, mathEnd);
    children.push({
      type: 'inlineMath',
      value: mathValue,
      html: captionMathHtml(mathValue),
      typst: convertLatex(mathValue),
    });
    index = mathEnd + 1;
  }

  return { type: 'caption', children: [{ type: 'paragraph', children }] };
}

function restoreIndependentGroupedTables(tree, utils, source) {
  utils.selectAll('container', tree).forEach((container) => {
    if (
      container.kind !== 'table' ||
      !Array.isArray(container.children) ||
      !container.position ||
      typeof source !== 'string'
    )
      return;

    const groupedTables = container.children.filter(
      (child) =>
        child?.type === 'container' &&
        child.kind === 'table' &&
        child.subcontainer === true,
    );
    if (groupedTables.length < 2) return;

    const sourceSlice = source.slice(
      container.position.start.offset,
      container.position.end.offset,
    );
    const captions = commandArguments(sourceSlice, 'caption');
    const labels = commandArguments(sourceSlice, 'label');
    const tables = groupedTables.map((group) =>
      group.children?.find((child) => child?.type === 'table'),
    );
    if (
      captions.length !== groupedTables.length ||
      labels.length !== groupedTables.length ||
      tables.some((table) => !table)
    )
      return;

    const gridChildren = [
      { type: 'raw', typst: '#grid(columns: 2, gutter: 8pt,\n' },
    ];
    const hiddenTargets = [];
    tables.forEach((table, index) => {
      const caption = captionFromLatex(captions[index]);
      const label = labels[index];

      if (index > 0) {
        // MyST's native thematic break reliably renders in the web app. Hide
        // it from Typst inside a comment because the PDF keeps these tables
        // side by side and therefore needs no horizontal separator.
        gridChildren.push(
          { type: 'raw', typst: '/*\n' },
          { type: 'thematicBreak' },
          { type: 'raw', typst: '*/\n' },
        );
      }

      gridChildren.push(
        { type: 'raw', typst: '[\n#figure([\n' },
        table,
        { type: 'raw', typst: '], caption: [\n' },
        ...caption.children,
        {
          type: 'raw',
          typst: `\n], kind: "table", supplement: [Table]) <${label}>\n],\n`,
        },
      );
      hiddenTargets.push({
        type: 'container',
        kind: 'table',
        label,
        identifier: label,
        children: [],
      });
    });
    gridChildren.push({ type: 'raw', typst: ')\n' });

    // The TeX has two independently captioned tables inside minipages. MyST
    // otherwise turns them into subfigures (Table 3 and Table 3a). Render the
    // same side-by-side layout while retaining two independent counter targets.
    container.type = 'div';
    delete container.kind;
    delete container.label;
    delete container.identifier;
    container.children = [
      ...gridChildren,
      { type: 'raw', typst: '#metadata(none)\n', children: hiddenTargets },
    ];
  });
}

function restoreExplicitReferenceSyntax(tree, utils) {
  utils.selectAll('crossReference', tree).forEach((node) => {
    // tex-to-myst represents `\eqref` as an empty reference. Supplying the
    // literal TeX template prevents a mismatched label prefix/target kind from
    // turning `equation~\eqref{...}` into e.g. “equation Table 1”.
    if (!node.children?.length)
      node.children = [{ type: 'text', value: '(%s)' }];
  });
}

const plugin = {
  name: 'MyST LaTeX compatibility',
  transforms: [
    {
      name: 'typst-cases-compatibility',
      stage: 'document',
      plugin: (_, utils) => (tree, file) => {
        let source;
        try {
          source = file?.path ? readFileSync(file.path, 'utf8') : undefined;
        } catch {
          source = undefined;
        }
        restoreExplicitReferenceSyntax(tree, utils);
        restoreLatexEquationNumbering(tree, utils, source);
        restoreLatexHeadingNumbering(tree, utils, source);
        restoreIndependentGroupedTables(tree, utils, source);
        restoreTableSectionHeaders(tree, utils);
        restoreNonFloatingProofEnvironments(tree, utils, source);
        restoreSubfigureCaptions(tree, utils);

        for (const type of ['math', 'inlineMath']) {
          utils.selectAll(type, tree).forEach((node) => {
            if (typeof node.value !== 'string') return;

            // MyST's Typst converter knows the named `\Vert` command, but its
            // equivalent TeX shorthand `\|` currently falls through as `|`.
            node.value = node.value.replaceAll(DOUBLE_BAR, DOUBLE_BAR_NAMED);

            // TeX's `\!` is a small -3mu adjustment (about -1/6 em), but the
            // Typst converter emits `#h(-1em)`. That exaggerated shift makes
            // adjacent symbols overlap, so prefer Typst's normal math spacing.
            node.value = node.value.replaceAll(NEGATIVE_THIN_SPACE, '');

            if (!node.value.includes(CASES_BEGIN)) return;
            const typst = convertMathWithCases(node.value);
            if (typst) node.typst = typst;
          });
        }
      },
    },
  ],
};

export default plugin;
