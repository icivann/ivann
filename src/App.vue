<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<script>
import { Vue } from 'vue-property-decorator';

export default class App extends Vue {
}
</script>

<style>
#nav a {
  font-weight: bold;
  color: #2c3e50;
}
#nav a.router-link-exact-active {
  color: #42b983;
}
</style>
