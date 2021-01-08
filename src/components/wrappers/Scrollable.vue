<template>
  <div class="scrollable">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Vue, Component } from 'vue-property-decorator';

@Component
export default class Scrollable extends Vue {
}
</script>

<style scoped>
  .scrollable {
    height: 100%;
    overflow: auto;
    scrollbar-width: none; /* For Mozilla */
  }

  ::-webkit-scrollbar {
    width: 0;
    background: none;
  }
</style>
