import { SITE_TITLE } from '@/data/site';

export function buildPageTitle(pageTitle: string): string {
  const trimmed = pageTitle.trim();
  return trimmed === '' ? SITE_TITLE : `${trimmed} | ${SITE_TITLE}`;
}
