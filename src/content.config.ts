import { defineCollection } from "astro:content";
import { z } from "astro/zod";

const blog = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.string(),
    updated: z.string(),
    tags: z.array(z.string()).min(1),
    categories: z.array(z.string()).min(1),
    permalink: z.string(),
    math: z.boolean(),
    draft: z.boolean(),
  }),
});

export const collections = { blog };
