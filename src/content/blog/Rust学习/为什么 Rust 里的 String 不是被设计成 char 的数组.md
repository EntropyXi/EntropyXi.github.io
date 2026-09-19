---
title: "为什么 Rust 里的 String 不是被设计成 char 的数组"
description: "从 C 语言的字符串索引出发，说明 Rust 的 String 为什么不实现 s[i]：UTF-8 变长编码让字节索引和字符索引都不成立，而 Rust 选择用 chars() 把解码成本与语义显式暴露给调用者。"
date: "2026-09-19T21:47:19+08:00"
updated: "2026-09-19T21:47:19+08:00"
tags:
  - "Rust"
  - "字符串"
  - "UTF-8"
  - "零成本抽象"
categories:
  - "Rust学习"
permalink: "2026/09/19/Rust学习/为什么Rust里的String不是被设计成char的数组"
math: false
draft: false
---

## 从 C 的哲学出发

我们是否有考虑过以下问题：在 C/C++ 类语言中，把 `String` 对象简单看作 `char` 数组，那么当我们在取值时，譬如 `s[0]`、`s[1]`，0 和 1 的索引到底代表着什么呢？

如果是表达第 0 个、第 1 个字节，这在 ASCII 码时代自然是合理的，譬如：

```rust
let s = String::from("hello");
```

每个英文字母的编码正好占着一个字节的位置。但是 Rust 的 `String` 采用 UTF-8 编码，如果是中文：

```rust
let s = String::from("你好");
```

UTF-8 中 你 -> 3 bytes，好 -> 3 bytes。所以内存更接近：

```text
你       好
E4 BD A0 E5 A5 BD
```

一共六个字节而不是两个，所以字节的表达不大正确。

那如果是表达第 0 个 Unicode 中的 char 呢？那么 `s[0]` 就得到 `你`，似乎很合理。但是这里又出现第二个问题：我们是怎么找到第 `i` 个字符的？例如 `s[1000]`，UTF-8 字符长度不固定，你无法像数组那样 `address = start + 1000 * sizeof(char)`，因为第一个字符可能占一个字节，第二个字符可能 3 bytes……这种查找复杂度是 `O(n)`，不符合我们直观的数组索引，所以也不可能。

## 理解 Rust 的思想

理解这个问题的关键是理解 Rust **显式大于隐式**的设计哲学：Rust 不愿意替你决定 0 到底指 byte、Unicode char 还是人类字符。

从底层所有权的角度看，`String` 在概念上非常接近：

```rust
struct String {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}
```

所以实际上，`String` 本质上就是拥有所有权的 UTF-8 `Vec<u8>`。但是区别在于，`Vec<u8>` 里可以装任何 type，而 `String` 这个 “`Vec<u8>`”，必须始终保证内部 byte 序列是合法 UTF-8。

这其实也体现了 Rust 的**类型系统思想**。Rust 很喜欢数据与类型双向约束的设计：**如果一个数据具有某种约束，就让类型保证这个约束**。

所以我们在字符串切片时也会有这个限制，例如：

```rust
let s = String::from("hello");
let a = &s[0..2];
```

得到 `he` 合法，但是：

```rust
let s = String::from("你好");
let a = &s[0..1];
```

会 panic。因为 `"你"` 是 `E4 BD A0`，而 `0..1` 只截到了 `E4`：这不是合法 UTF-8，所以这个切片就是不合法的。

更直观地说，每当你想说 “我要字符串里的第 i 个元素”，Rust 都会反问：什么叫元素？Rust 认为，程序员必须显式地明确表达自己的意图。这其实和 Rust 的零成本抽象也有关系。简单理解就是 Rust 认为高级抽象不应该迫使你支付不必要的性能成本。假设 Rust 把 `String` 实现为 `Vec<char>`，那么英文 `hello` 需要 20 个字节来存储，而使用 UTF-8，我们只需要用 5 个字节。Rust 正是不愿意为了让 `s[i]` 方便一点而让所有程序永久承担 4 倍左右的某些文本内存成本。

## Rust 推荐的方式

Rust 推荐使用 `chars()` 方法：

```rust
let s = String::from("你好");

for c in s.chars() {
    println!("{c}");
}
```

Rust 明确告诉你他正在进行 UTF-8 解码。也就是说 `s.chars()` 这个 API 本身就在表达成本和语义。而如果 Rust 允许 `s[i]` 的话，你看代码根本不知道：这是 byte 索引？Unicode 索引？是否进行了 UTF-8 解码？复杂度是 `O(1)` 还是 `O(n)`？而 Rust 选择让这些东西显式。
