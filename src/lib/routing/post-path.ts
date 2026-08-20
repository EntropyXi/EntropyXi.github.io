export function buildPostPath(permalink: string): string {
  const encoded = permalink
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `/${encoded}/`;
}
