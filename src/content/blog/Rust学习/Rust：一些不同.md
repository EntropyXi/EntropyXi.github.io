---
title: "Rust：一些不同"
description: "Rust 默认变量不可变，需要 mut 才能重新赋值，而 const 不允许与 mut 同时使用。顺带梳理变量遮蔽与 mut 的区别，以及遮蔽为什么能改变值的类型。"
date: "2026-09-18T11:10:49+08:00"
updated: "2026-09-18T11:10:49+08:00"
tags:
  - "Rust"
  - "可变性"
  - "变量遮蔽"
categories:
  - "Rust学习"
permalink: "2026/09/18/Rust学习/Rust：一些不同"
math: false
draft: false
---

在 Rust 中，默认地，声明变量并赋值后，便不能改变这个变量的值，如：

```rust
fn main() {
    let x = 5;
    println!("{x}");
    x = 6;
    println!("{x}");
}
```

执行这段语句时，会报错 `cannot assign twice to immutable variable x`。因为我们试图将不可变的、已经赋值好的 `x` 从 `5` 变成 `6`。我们得通过在变量名前添加 `mut` 来使得它们可变：

```rust
fn main() {
    let mut x = 5;
    println!("{x}");
    x = 6;
    println!("{x}");
}
```

但声明常量的关键字 `const` 不允许和 `mut` 一起使用。

## 遮蔽

在 Rust 中，我们可以声明一个与先前变量名相同的变量。这意味着，先声明的变量被后声明的变量 **遮蔽** 了。第二个变量遮蔽了第一个变量，将该变量名的所有使用都归于自己，直到它自身被遮蔽或作用域结束：

```rust
fn main() {
    let x = 5;
    let x = x + 1;
    {
        let x = x * 2;
        println!("The value of x in the inner scope is: {x}");
    }
    println!("The value of x is: {x}");
}
```

输出：

```bash
$ cargo run
   Compiling variables v0.1.0 (file:///projects/variables)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s
     Running `target/debug/variables`
The value of x in the inner scope is: 12
The value of x is: 6
```

`mut` 与遮蔽的另一个区别是，因为当我们再次使用 `let` 关键字时，实际上是在创建一个新变量，所以我们可以改变值的类型：

```rust
let spaces = " ";
let spaces = spaces.len();
```

是正确的。而：

```rust
let mut spaces = " ";
spaces = spaces.len();
```

会报错。
