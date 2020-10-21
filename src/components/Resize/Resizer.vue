<template>
  <div class="resizer">
    <slot></slot>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';

@Component({})
export default class Resizer extends Vue {
}
</script>

<style>
  .resizer {
    display: block;
    position: relative;
    z-index: 2;
    width: 10px;
    height: 100%;
    margin-left: -10px;
    left: 5px;
    cursor: ew-resize;
  }
</style>
