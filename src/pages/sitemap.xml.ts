import { getCollection } from "astro:content";
import { SITE_URL } from "@/data/site";
import { sortPostsByDateDesc } from "@/lib/content/aggregate";
import { buildPostPath } from "@/lib/routing/post-path";

export async function GET(): Promise<Response> {
  const posts = sortPostsByDateDesc(
    await getCollection("blog", ({ data }) => data.draft !== true),
  );
  const staticPaths = [
    "/",
    "/about/",
    "/archives/",
    "/categories/",
    "/tags/",
    "/search/",
  ];
  const postPaths = posts.map((post) => buildPostPath(post.data.permalink));
  const urls = [...staticPaths, ...postPaths]
    .map((path) => `  <url><loc>${new URL(path, SITE_URL).href}</loc></url>`)
    .join("\n");

  const body = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls}
</urlset>
`;

  return new Response(body, {
    headers: { "Content-Type": "application/xml; charset=utf-8" },
  });
}
