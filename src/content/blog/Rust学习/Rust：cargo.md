---
title: "Rust：cargo"
description: "cargo 是 Rust 的构建系统与包管理器。记录 cargo new 生成的项目结构、Cargo.toml 与 Cargo.lock 各自的职责，以及 build、run、check 与 --release 构建之间的区别。"
date: "2026-09-18T00:11:54+08:00"
updated: "2026-09-18T00:11:54+08:00"
tags:
  - "Rust"
  - "Cargo"
  - "构建工具"
categories:
  - "Rust学习"
permalink: "2026/09/18/Rust学习/Rust：cargo"
math: false
draft: false
---

cargo 是 Rust 的构建系统与包管理器。

用 cargo 创建项目：

```bash
cargo new hello_world
```

会新建一个名为 `hello_world` 的文件夹，里面自动创建好了 `.git` 文件夹、`src` 目录及 `main.rs`，和 `Cargo.toml` 文件。

## Cargo.toml

```toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2024"

[dependencies]
```

`package` 标明接下来是在构建一个包，`dependencies` 在记录这个项目需要的依赖项/包（crate）。

## 使用 cargo 构建并运行

```bash
cargo build
```

然后默认以调试构建输出在 `target/debug/` 里：

```text
.\target\debug\hello_world.exe
```

即可运行。或者一条命令：

```bash
cargo run
```

首次运行 `cargo build` 还会导致 cargo 在顶层创建一个新文件：`Cargo.lock`。此文件跟踪项目中依赖项的确切版本，cargo 会自动管理此内容。

此外：

```bash
cargo check
```

还可以在不生成可执行文件的前提下对代码进行检查，确保其能够被编译。一个理想的编程习惯是在编写程序时定期运行 `cargo check`。

## 为发布构建

当你的代码准备好发布成一个正式的项目时，可以使用 `cargo build --release` 来编译并启用优化。这个命令会在 `target/release` 而不是 `target/debug` 中生成可执行文件。优化能让你的 Rust 代码运行得更快，但开启优化会延长程序编译所需的时间。
