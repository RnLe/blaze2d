/** Convert Rust UTF-8 source spans to the UTF-16 positions used by CodeMirror. */
export function editorSpan(source: string, span: [number, number]): [number, number] {
  let bytes = 0, units = 0, from = 0, to = 0;
  for (const character of source) {
    const length = new TextEncoder().encode(character).length;
    if (bytes + length <= span[0]) from = units + character.length;
    if (bytes < span[1]) to = units + character.length;
    units += character.length; bytes += length;
  }
  return [Math.min(from, source.length), Math.min(Math.max(to, from), source.length)];
}
