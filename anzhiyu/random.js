var posts=["2026/02/08/hello-world/","2026/02/08/深度学习/RMSProp/","2026/02/08/深度学习/CNN/CNN的设计原则与数学推导/"];function toRandomPost(){
    pjax.loadUrl('/'+posts[Math.floor(Math.random() * posts.length)]);
  };