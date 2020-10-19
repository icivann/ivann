<template>
  <div class="button-grid">
    <slot>Add your buttons here.</slot>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';

@Component({})
export default class ButtonGrid extends Vue {
}
</script>

<style scoped>

</style>
