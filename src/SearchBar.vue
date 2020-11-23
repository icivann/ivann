<template>
  <div class="box">
    <i class="fa fa-search" aria-hidden="true"/>
    <input
      ref="search"
      tabindex="0"
      @input="$emit('value-change', $event.target.value)"
      spellcheck="false"
      autocomplete="off"
      maxlength="20"
    >
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';

@Component
export default class SearchBar extends Vue {
}
</script>

<style scoped>
.box {
  background: var(--foreground);
  height: 2rem;
  width: 100%;
  border-radius: 6px;
  transition-duration: 0.1s;
  position: relative;
}
.box:hover {
  background: #e0e0e0;
  transition-duration: 0.1s;
}
.fa-search {
  color: var(--background);
  z-index: 10;
  position: absolute;
  margin-left: 8px;
  margin-top: 0.5rem;
}
input {
  background: none;
  color: var(--background);
  border: none;
  width: 100%;
  padding: 0 1rem 4px 1.7rem;
  height: 100%;
  cursor: default;
}
input::selection {
  background: var(--dark-grey);
  color: var(--foreground);
}
textarea:focus, input:focus {
  outline: none;
  cursor: text;
}
</style>
