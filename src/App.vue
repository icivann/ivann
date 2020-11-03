<template>
  <div id="app">
    <router-view/>
    <cookie-law theme="dark-lime"></cookie-law>
  </div>
</template>

<script>
import CookieLaw from 'vue-cookie-law-with-type';
import { Component, Vue } from 'vue-property-decorator';

@Component({
  components: {
    CookieLaw,
  },
})
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

.Cookie {
  background: var(--grey) !important;
  padding-top: 0.5em !important;
  padding-bottom: 0.5em !important;
}

.Cookie__button {
  background: var(--blue) !important;
  padding-top: 0.2em !important;
  padding-bottom: 0.2em !important;
}
</style>
