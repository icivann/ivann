export function download(label: string, content: string) {
  const blob = new Blob([content], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = window.URL.createObjectURL(blob);
  a.download = label;
  a.click();
}

export function downloadPython(label: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = window.URL.createObjectURL(blob);
  a.download = `${label}.py`;
  a.click();
}
