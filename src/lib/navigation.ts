export interface NavItem {
  label: string;
  href: string;
}

export const NAV_ITEMS: NavItem[] = [
  { label: "首页", href: "/" },
  { label: "归档", href: "/archives/" },
  { label: "分类", href: "/categories/" },
  { label: "标签", href: "/tags/" },
  { label: "关于", href: "/about/" },
  { label: "搜索", href: "/search/" },
];

export function isNavActive(href: string, currentPath: string): boolean {
  if (href === "/") return currentPath === "/";
  return currentPath.startsWith(href);
}
