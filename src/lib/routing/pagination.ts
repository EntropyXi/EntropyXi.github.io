export const POSTS_PER_PAGE = 8;

export function paginate<T>(
  items: T[],
  page: number,
  perPage = POSTS_PER_PAGE,
): T[] {
  const start = (page - 1) * perPage;
  return items.slice(start, start + perPage);
}

export function totalPages(
  totalItems: number,
  perPage = POSTS_PER_PAGE,
): number {
  return Math.max(1, Math.ceil(totalItems / perPage));
}

export function pagePath(page: number): string {
  return page === 1 ? "/" : `/page/${page}/`;
}
