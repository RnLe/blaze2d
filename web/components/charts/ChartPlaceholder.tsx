/** Reserves a chart's box while its recorded dataset loads. */
export default function ChartPlaceholder({
  width,
  height,
  label = 'Loading benchmark data…',
}: {
  width: number;
  height: number;
  label?: string;
}) {
  return (
    <div className="chart-placeholder" style={{ maxWidth: width, height }} role="status">
      {label}
    </div>
  );
}
