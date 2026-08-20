import { getCollection } from "astro:content";
import { SITE_DESCRIPTION, SITE_TITLE, SITE_URL } from "@/data/site";
import { sortPostsByDateDesc } from "@/lib/content/aggregate";
import { buildPostPath } from "@/lib/routing/post-path";

export async function GET(): Promise<Response> {
  const posts = sortPostsByDateDesc(
    await getCollection("blog", ({ data }) => data.draft !== true),
  );
  const entries = posts
    .map((post) => {
      const href = new URL(buildPostPath(post.data.permalink), SITE_URL).href;
      return `  <entry>
    <title>${post.data.title}</title>
    <id>${href}</id>
    <link href="${href}" />
    <published>${post.data.date}</published>
    <updated>${post.data.updated}</updated>
    <summary>${post.data.description}</summary>
  </entry>`;
    })
    .join("\n");

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>${SITE_TITLE}</title>
  <subtitle>${SITE_DESCRIPTION}</subtitle>
  <id>${SITE_URL}/</id>
  <link href="${SITE_URL}/" />
  <updated>${posts[0]?.data.updated ?? "2026-01-01T00:00:00Z"}</updated>
${entries}
</feed>
`;

  return new Response(body, {
    headers: { "Content-Type": "application/atom+xml; charset=utf-8" },
  });
}
