// D:\Blog_file\scripts\mathjax.js

// 这是一个脚本注入器，它会自动把 MathJax 的代码插入到每个页面的底部
hexo.extend.injector.register('body_end', `
  <script src="//cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
  <script>
    MathJax.Hub.Config({
      tex2jax: {
        inlineMath: [ ['$','$'], ["\\(","\\)"] ],
        processEscapes: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    });
  </script>
`);