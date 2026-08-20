import type { CollectionEntry } from "astro:content";
import { totalPages } from "@/lib/routing/pagination";

export type BlogPost = CollectionEntry<"blog">;

export function sortPostsByDateDesc(posts: BlogPost[]): BlogPost[] {
  return posts.sort((a, b) => {
    const dateA = new Date(a.data.date).getTime();
    const dateB = new Date(b.data.date).getTime();
    if (dateB !== dateA) return dateB - dateA;
    return b.data.permalink.localeCompare(a.data.permalink);
  });
}

export function postYear(post: BlogPost): string {
  return post.data.date.slice(0, 4);
}

export function postMonth(post: BlogPost): string {
  return post.data.date.slice(5, 7);
}

export function groupByYearMonth(
  posts: BlogPost[],
): Map<string, Map<string, BlogPost[]>> {
  const years = new Map<string, Map<string, BlogPost[]>>();
  for (const post of posts) {
    const year = postYear(post);
    const month = postMonth(post);
    const months = years.get(year) ?? new Map<string, BlogPost[]>();
    const list = months.get(month) ?? [];
    list.push(post);
    months.set(month, list);
    years.set(year, months);
  }
  return years;
}

export function postsInYear(posts: BlogPost[], year: string): BlogPost[] {
  return posts.filter((post) => postYear(post) === year);
}

export function postsInYearMonth(
  posts: BlogPost[],
  year: string,
  month: string,
): BlogPost[] {
  return posts.filter(
    (post) => postYear(post) === year && postMonth(post) === month,
  );
}

export function postsInCategory(
  posts: BlogPost[],
  category: string[],
): BlogPost[] {
  return posts.filter((post) => {
    const categories = post.data.categories;
    if (categories.length < category.length) return false;
    return category.every((segment, index) => categories[index] === segment);
  });
}

export function postsWithTag(posts: BlogPost[], tag: string): BlogPost[] {
  return posts.filter((post) => post.data.tags.includes(tag));
}

export function categoryPageCount(posts: BlogPost[], perPage: number): number {
  return totalPages(posts.length, perPage);
}

export function tagPageCount(posts: BlogPost[], perPage: number): number {
  return totalPages(posts.length, perPage);
}

export function buildCategoryPath(category: string[]): string {
  return `/categories/${category.map((segment) => encodeURIComponent(segment)).join("/")}/`;
}

export function buildTagPath(tag: string): string {
  return `/tags/${encodeURIComponent(tag)}/`;
}

export function buildArchivePath(year?: string, month?: string): string {
  if (year === undefined) return "/archives/";
  if (month === undefined) return `/archives/${year}/`;
  return `/archives/${year}/${month}/`;
}
