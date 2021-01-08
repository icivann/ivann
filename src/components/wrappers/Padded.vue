<template>
  <div class="padded">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Vue, Component } from 'vue-property-decorator';

@Component
export default class Padded extends Vue {
}
</script>

<style scoped>
  .padded {
    padding: 1rem;
  }
</style>
