<template>
  <div id="app">
    <div id="nav">
      <router-link to="/">Home</router-link> |
      <router-link to="/about">About</router-link>
    </div>
    <router-view/>
    <baklava-editor :plugin="viewPlugin"></baklava-editor>
  </div>
</template>

<script>
import { Editor } from '@baklavajs/core';
import { ViewPlugin } from '@baklavajs/plugin-renderer-vue';

export default {
  data() {
    return {
      editor: new Editor(),
      viewPlugin: new ViewPlugin(),
    };
  },
  created() {
    this.editor.use(this.viewPlugin);
    // register your nodes, node options, node interface types, ...
  },
};
</script>

<style>
#app {
  width: 2000px;
  height: 1000px;
}

#nav {
  padding: 30px;
}
#nav a {
  font-weight: bold;
  color: #2c3e50;
}
#nav a.router-link-exact-active {
  color: #42b983;
}
</style>
