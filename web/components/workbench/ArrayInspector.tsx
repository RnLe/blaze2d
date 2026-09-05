'use client';
import { useState } from 'react';
import type { Arrays, NumericArray } from '../../lib/contract/records';

function number(value: number) { return value === 0 ? '0' : value.toPrecision(5); }
function cell(array: NumericArray, index: number) {
  if (array.dtype === 'float64') return number(array.data[index]);
  const real = array.data[2 * index], imaginary = array.data[2 * index + 1];
  return `${number(real)} ${imaginary < 0 ? '−' : '+'} ${number(Math.abs(imaginary))}i`;
}

export function ArrayInspector({ arrays }: { arrays: Arrays }) {
  const names = Object.keys(arrays), [selected, setSelected] = useState(''), [indices, setIndices] = useState<number[]>([]);
  const [rowPage, setRowPage] = useState(0), [columnPage, setColumnPage] = useState(0);
  const name = arrays[selected] ? selected : names[0], array = arrays[name];
  if (!array) return <p>No arrays retained in this record.</p>;
  const shape = array.shape, rank = shape.length;
  const rows = rank > 1 ? shape[rank - 2] : shape[0] ?? 1, columns = rank > 1 ? shape[rank - 1] : 1;
  let prefix = 0;
  for (let i = 0; i < rank - 2; i++) prefix = prefix * shape[i] + Math.max(0, Math.min(indices[i] ?? 0, shape[i] - 1));
  const rowStart = Math.min(rowPage * 20, Math.max(0, rows - 1)), columnStart = Math.min(columnPage * 12, Math.max(0, columns - 1));
  const visibleRows = Math.min(20, rows - rowStart), visibleColumns = Math.min(12, columns - columnStart);
  return <section className="wb-inspector">
    <label className="wb-field"><span>Dataset</span><select value={name} onChange={event => { setSelected(event.target.value); setIndices([]); setRowPage(0); setColumnPage(0); }}>
      {names.map(name => <option key={name} value={name}>{name}</option>)}</select></label>
    <p className="wb-array-shape">{array.dtype} · ({shape.join(', ')}) · {array.dimensions.join(' × ')} · C order</p>
    {rank > 2 && <div className="wb-fields">{shape.slice(0, -2).map((size, index) => <label className="wb-field" key={index}><span>{array.dimensions[index]} index</span>
      <input type="number" min={0} max={size - 1} value={indices[index] ?? 0} onChange={event => setIndices(previous => {
        const next = [...previous]; next[index] = Math.min(size - 1, Math.max(0, Number(event.target.value))); return next;
      })} /></label>)}</div>}
    <div className="wb-table-scroll" tabIndex={0} role="region" aria-label={`${name} array values`}>
      <table><thead><tr><th scope="col">{array.dimensions[Math.max(0, rank - 2)]}</th>{Array.from({ length: visibleColumns }, (_, column) => <th scope="col" key={column}>{rank > 1 ? columnStart + column : 'value'}</th>)}</tr></thead>
        <tbody>{Array.from({ length: visibleRows }, (_, row) => <tr key={row}><th scope="row">{rowStart + row}</th>{Array.from({ length: visibleColumns }, (_, column) =>
          <td key={column}>{cell(array, (prefix * rows + rowStart + row) * columns + columnStart + column)}</td>)}</tr>)}</tbody>
      </table>
    </div>
    {(rows > 20 || columns > 12) && <div className="wb-row">
      <button disabled={rowPage === 0} onClick={() => setRowPage(page => page - 1)}>Previous rows</button>
      <button disabled={rowStart + visibleRows >= rows} onClick={() => setRowPage(page => page + 1)}>Next rows</button>
      {columns > 12 && <><button disabled={columnPage === 0} onClick={() => setColumnPage(page => page - 1)}>Previous columns</button>
        <button disabled={columnStart + visibleColumns >= columns} onClick={() => setColumnPage(page => page + 1)}>Next columns</button></>}
    </div>}
  </section>;
}
