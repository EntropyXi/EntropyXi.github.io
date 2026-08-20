// 旧站 `/search.xml` 的兼容入口。迁移期输出空索引，阶段 4 由 Pagefind
// 提供新搜索；该 URL 只保证旧外部引用不 404。
export function GET(): Response {
  const body = `<?xml version="1.0" encoding="UTF-8"?>
<search>
</search>
`;

  return new Response(body, {
    headers: { 'Content-Type': 'application/xml; charset=utf-8' },
  });
}
