export function download(label: string, content: string) {
  const blob = new Blob([content], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = window.URL.createObjectURL(blob);
  a.download = label;
  a.click();
}
