import { SITE_DESCRIPTION, SITE_TITLE, SITE_URL } from '@/data/site';

export function GET(): Response {
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>${SITE_TITLE}</title>
  <subtitle>${SITE_DESCRIPTION}</subtitle>
  <id>${SITE_URL}/</id>
  <link href="${SITE_URL}/" />
  <updated>2026-01-01T00:00:00Z</updated>
</feed>
`;

  return new Response(body, {
    headers: { 'Content-Type': 'application/atom+xml; charset=utf-8' },
  });
}
