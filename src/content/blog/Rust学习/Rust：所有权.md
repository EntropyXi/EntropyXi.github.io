---
title: "Rust：所有权"
description: "所有权是 Rust 在编译期管理内存的一组规则。整理三条所有权规则、String 在栈与堆上的内存布局，以及移动、克隆和作用域赋值背后的双重释放问题。"
date: "2026-09-18T13:57:00+08:00"
updated: "2026-09-18T13:57:00+08:00"
tags:
  - "Rust"
  - "所有权"
  - "内存管理"
categories:
  - "Rust学习"
permalink: "2026/09/18/Rust学习/Rust：所有权"
math: false
draft: false
---

所有权是管理 Rust 程序如何管理内存的一组规则。有些语言需要手动管理内存的分配与释放；有些则通过垃圾回收机制定期查找回收不再使用的内存。而 Rust 采用了第三种方式：内存通过所有权系统来管理，该系统拥有一组规则，由编译器进行检查。如果违反了任何规则，程序将无法编译。

## 所有权规则

- Rust 中的每一个值都有一个**所有者**
- 同一时间只能有一个**所有者**
- 当**所有者**离开作用域时，该值将被丢弃

## String 类型

直接介绍的数据类型的大小都是已知的，我们可以简单地将他们存储在栈里，并在其作用域结束时弹出。但我们想要研究存储在堆上的数据，并观察 Rust 如何知道何时清理这些数据，`String` 类型就是很好的例子。

字符串字面量被硬编码在变量上，但不是每时每刻我们都知道这个变量存储的内容是什么的，比如我们去获取命令行的输入。这时候我们就需要 `String`：

```rust
let s = String::from("hello");
```

这种字符串内容本身可修改：

```rust
s.push_str(", world!");
println!("{s}");
```

那么，这里的区别是什么呢？为什么 `String` 可以被修改，而字符串字面量却不行？区别在于这两种类型处理内存的方式不同。

## 内存与分配

```rust
{
    let s = String::from("hello");
}
```

在我们创建 `String` 对象时，会自动地调用内存分配器为新建的实例在堆上请求所需的内存。当离开作用域时，Rust 会为我们调用一个特殊的函数。这个函数叫做 `drop`，`String` 的作者可以在这里放置归还内存的代码。Rust 会在右花括号处自动调用 `drop`。

现在看起来似乎很简单，但当我们需要管理多个内存分配在堆上的变量实例时，代码的行为可能会出乎意料，下面我们来探讨其中一些情况。

### 移动

```rust
let s1 = String::from("hello");
let s2 = s1;
```

直觉告诉我们第二行会复制 `s1` 中的值并将其绑定到 `s2`。但实际情况并非如此。

一个 `String` 由左侧三部分组成：一个指向存放字符串内容的内存的指针、一个长度和一个容量。这组数据存储在栈上。右侧是堆上存放内容的内存。

![图 4-1：String 在内存中的表示，将值 "hello" 绑定到 s1；栈上的 ptr、len 与 capacity 指向堆上的字符序列](/images/rust/rust-string-in-memory-s1.png)

当我们将 `s1` 赋值给 `s2` 时，`String` 的数据被复制，这意味着我们复制了栈上的指针、长度和容量。我们并未复制指针所指向的堆上的数据。

![图 4-2：变量在内存中的表示，s2 拥有 s1 的指针、长度和容量的副本，两个指针指向同一块堆内存](/images/rust/rust-string-move-s2.png)

之前我们说过，当变量离开作用域时，Rust 会自动调用 `drop` 函数并清理该变量对应的堆内存。但上图显示两个数据指针都指向同一位置。这就有问题了：当 `s2` 和 `s1` 离开作用域时，它们都会尝试释放同一块内存。这被称为**双重释放错误**。

为确保内存安全，在执行 `let s2 = s1;` 这一行之后，Rust 认为 `s1` 不再有效。因此，当 `s1` 离开作用域时，Rust 不需要释放任何东西。

```rust
    let s1 = String::from("hello");
    let s2 = s1;

    println!("{s1}, world!");
```

我们会遇到这样的错误：

```bash
$ cargo run
   Compiling ownership v0.1.0 (file:///projects/ownership)
error[E0382]: borrow of moved value: `s1`
 --> src/main.rs:5:16
  |
2 |     let s1 = String::from("hello");
  |         -- move occurs because `s1` has type `String`, which does not implement the `Copy` trait
3 |     let s2 = s1;
  |              -- value moved here
4 |
5 |     println!("{s1}, world!");
  |                ^^ value borrowed here after move
  |
  = note: this error originates in the macro `$crate::format_args_nl` which comes from the expansion of the macro `println` (in Nightly builds, run with -Z macro-backtrace for more info)
help: consider cloning the value if the performance cost is acceptable
  |
3 |     let s2 = s1.clone();
  |                ++++++++

For more information about this error, try `rustc --explain E0382`.
error: could not compile `ownership` (bin "ownership") due to 1 previous error
```

这种复制指针、长度和容量而不复制数据的概念听起来像是在浅拷贝。但由于 Rust 还会使第一个变量失效，所以它不叫浅拷贝，而是被称为**移动**。

![图 4-4：s1 失效后内存中的表示，只有 s2 仍然有效并指向堆数据](/images/rust/rust-string-move-s1-invalid.png)

所以只有 `s2` 有效时，当它离开作用域时，它将独自释放内存。

此外，这还隐含了一个设计选择：Rust 永远不会自动创建数据的“深”拷贝。因此，任何自动拷贝都可以被认为在运行时性能方面是低开销的。

### 作用域与赋值

简单理解：

```rust
    let mut s = String::from("hello");
    s = String::from("ahoy");

    println!("{s}, world!");
```

### 克隆

如果我们确实想要深度复制 `String` 的堆数据，而不仅仅是栈数据，我们可以使用一个名为 `clone` 的方法：

```rust
    let s1 = String::from("hello");
    let s2 = s1.clone();

    println!("s1 = {s1}, s2 = {s2}");
```

代码可以正常工作，堆数据确实被复制了。
